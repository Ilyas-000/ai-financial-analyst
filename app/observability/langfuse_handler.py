"""Langfuse callback factory and per-request trace attributes.

Single point of integration for I-10. ``ChatService`` attaches the callback
returned by :func:`get_langfuse_callback` to the ``RunnableConfig`` it passes
into the LangGraph graph; LangChain then propagates the callback down to every
nested LLM / tool / chain runnable, so we don't have to touch each node.

Design notes:

* In Langfuse v4 the SDK separates "client" (HTTP / OTel exporter, lives once
  per process) from "callback handler" (a thin LangChain bridge that talks to
  the active client via ``get_client()``). We construct the client lazily on
  first use; the handler itself becomes a no-op object once created — safe to
  share across requests.
* When ``enable_langfuse=False`` (the default per ``.env.example``) we never
  construct the client and ``get_langfuse_callback()`` returns ``None``;
  ``ChatService`` then simply omits the ``callbacks`` key from
  ``RunnableConfig``. No HTTP, no background threads, no env requirements.
* Per-request metadata (thread_id, tenant, role) is attached via Langfuse's
  ``propagate_attributes`` context manager — it stamps the surrounding OTel
  context, so every span emitted by the callback within the ``with`` block
  inherits ``session_id`` / ``user_id`` / ``metadata``. ``contextvars`` carry
  this state correctly across ``await`` boundaries, so a single sync ``with``
  in ``ChatService`` covers the whole async graph run / event stream.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.config import get_settings
from app.graph.state import UserRole

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _init_langfuse_client() -> Langfuse | None:
    """Construct the process-wide Langfuse client, or ``None`` if disabled.

    Cached so we pay the OTel setup cost exactly once per process. The client
    holds a background flush thread, so calling it many times would leak
    workers.
    """
    settings = get_settings()
    if not settings.enable_langfuse:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.warning(
            "ENABLE_LANGFUSE=true but LANGFUSE_PUBLIC_KEY/SECRET_KEY are empty; "
            "tracing will be skipped."
        )
        return None
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.app_env,
        tracing_enabled=True,
        debug=settings.langfuse_debug,
        timeout=settings.langfuse_timeout_seconds,
    )
    logger.info(
        "Langfuse client initialised against %s (debug=%s, timeout=%ss)",
        settings.langfuse_host,
        settings.langfuse_debug,
        settings.langfuse_timeout_seconds,
    )
    return client


@lru_cache(maxsize=1)
def get_langfuse_callback() -> CallbackHandler | None:
    """Return a shared Langfuse callback handler, or ``None`` if disabled.

    The same handler instance is reused across requests — it is stateless and
    relies on the active OTel context (set by :func:`langfuse_trace_attributes`)
    to correlate spans to a trace.
    """
    if _init_langfuse_client() is None:
        return None
    return CallbackHandler()


def shutdown_langfuse() -> None:
    """Flush the export queue on application shutdown.

    OTel's BatchSpanProcessor flushes on a 5s timer; without an explicit
    flush, the last spans of a request can sit in the buffer when the user
    Ctrl+C's the dev server. Called from the FastAPI lifespan's finally
    block. Idempotent: a no-op if the client was never constructed.
    """
    client = _init_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
        logger.info("Langfuse client flushed on shutdown")
    except Exception as exc:
        logger.warning("Langfuse flush failed during shutdown: %s", exc)


@contextmanager
def langfuse_trace_attributes(
    *,
    thread_id: str,
    user_role: UserRole,
    company_id: int,
) -> Iterator[None]:
    """Stamp the surrounding OTel context with per-request trace attributes.

    ``thread_id`` → ``session_id`` (groups multi-turn conversation in the UI);
    ``company_id`` → ``user_id`` (tenant-scoped identity for the demo, in
    production this would be the authenticated user id); ``user_role`` and
    ``company_id`` are also stored in metadata for filtering. No-op when
    Langfuse is disabled, so callers don't need a separate code path.
    """
    if _init_langfuse_client() is None:
        yield
        return
    with propagate_attributes(
        session_id=thread_id,
        user_id=f"tenant:{company_id}",
        metadata={
            "user_role": user_role,
            "company_id": str(company_id),
            "thread_id": thread_id,
        },
    ):
        yield
