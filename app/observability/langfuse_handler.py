"""Langfuse callback factory and per-request trace attributes (I-10).

``ChatService`` attaches the callback from :func:`get_langfuse_callback` to
the ``RunnableConfig``; LangChain propagates it to every nested runnable, so
nodes stay integration-agnostic.

Non-obvious bits:

* Langfuse v4 splits the process-wide "client" (HTTP/OTel exporter) from the
  per-request "callback handler". We build the client lazily and share the
  handler across requests.
* When ``enable_langfuse=False`` (default) the client is never built and
  ``get_langfuse_callback()`` returns ``None`` — no HTTP, threads, or env deps.
* Per-request metadata is stamped via ``propagate_attributes``; ``contextvars``
  carry it across ``await``, so one sync ``with`` covers the whole async run.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

import structlog
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.config import get_settings
from app.graph.state import UserRole

logger = structlog.get_logger(__name__)


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
        logger.warning("langfuse_enabled_but_keys_missing_skipping")
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
        "langfuse_client_initialised",
        host=settings.langfuse_host,
        debug=settings.langfuse_debug,
        timeout_seconds=settings.langfuse_timeout_seconds,
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
        logger.info("langfuse_flushed_on_shutdown")
    except Exception as exc:
        logger.warning("langfuse_flush_failed", error=str(exc))


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
