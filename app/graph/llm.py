"""LLM factory and a transient-retry wrapper.

Single source of truth for ``ChatOllama`` construction in I-06+ nodes
(Supervisor / DirectAnswer / Clarify / Finalize). Existing subgraphs keep
their own factories until a future cleanup pass.

``invoke_llm`` performs one bounded retry on transient errors
(timeout / connection / 5xx) with a short backoff. Anything that survives the
retry surfaces as ``LLMUnavailableError`` so the chat service can return a
graceful Russian-language degradation message instead of bubbling a 500.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal

import httpx
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from app.config import get_settings

logger = logging.getLogger(__name__)

LLMRole = Literal["supervisor", "specialist", "writer"]


class LLMUnavailableError(RuntimeError):
    """LLM service didn't respond after the bounded retry."""


def make_llm(role: LLMRole, *, temperature: float = 0.0) -> ChatOllama:
    settings = get_settings()
    model = {
        "supervisor": settings.llm_supervisor_model,
        "specialist": settings.llm_specialist_model,
        "writer": settings.llm_writer_model,
    }[role]
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model,
        temperature=temperature,
        timeout=settings.llm_request_timeout,
    )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    # langchain-ollama wraps low-level errors; fall back to message sniffing.
    message = str(exc).lower()
    return any(
        token in message
        for token in ("timeout", "timed out", "connection", "connect error", "refused")
    )


async def invoke_llm(llm: ChatOllama, messages: Sequence[BaseMessage]) -> BaseMessage:
    settings = get_settings()
    try:
        return await llm.ainvoke(list(messages))
    except Exception as exc:
        if not _is_transient(exc):
            raise
        logger.warning("llm transient error, retrying once: %s: %s", type(exc).__name__, exc)
        await asyncio.sleep(settings.llm_retry_backoff_seconds)
        try:
            return await llm.ainvoke(list(messages))
        except Exception as exc2:
            raise LLMUnavailableError(
                f"ollama unreachable after retry: {type(exc2).__name__}: {exc2}"
            ) from exc2
