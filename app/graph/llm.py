"""LLM factory and a transient-retry wrapper.

Single source of truth for ``ChatOllama`` construction across all graph
nodes — Supervisor / DirectAnswer / Clarify / Finalize / Condense plus the
``sql_analyst`` / ``docs_researcher`` subgraphs (unified in I-15, closing
TD-05). The answer-producing call passes ``tags=[FINAL_ANSWER_TAG]`` so its
streaming events surface to the UI.

UI plumbing (I-09): ``streaming=True`` makes ``astream_events`` yield
per-token chunks; ``tags=[FINAL_ANSWER_TAG]`` marks the one call whose tokens
Chainlit streams to the user (see ``chat_service.py`` / ``chainlit_app.py``).

``invoke_llm`` does one bounded retry on transient errors (timeout /
connection / 5xx); anything surviving it surfaces as ``LLMUnavailableError``
so the chat service degrades gracefully instead of 500-ing.
"""

import asyncio
from collections.abc import Sequence
from typing import Literal

import httpx
import structlog
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.config import get_settings

logger = structlog.get_logger(__name__)

LLMRole = Literal["supervisor", "specialist", "writer"]

LLMRunnable = Runnable[Sequence[BaseMessage], BaseMessage]

FINAL_ANSWER_TAG = "final_answer"


class LLMUnavailableError(RuntimeError):
    """LLM service didn't respond after the bounded retry."""


def make_llm(
    role: LLMRole,
    *,
    temperature: float = 0.0,
    tags: Sequence[str] | None = None,
) -> LLMRunnable:
    settings = get_settings()
    model = {
        "supervisor": settings.llm_supervisor_model,
        "specialist": settings.llm_specialist_model,
        "writer": settings.llm_writer_model,
    }[role]
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=model,
        temperature=temperature,
        timeout=settings.llm_request_timeout,
        streaming=True,
    )
    if tags:
        return llm.with_config({"tags": list(tags)})
    return llm


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


async def invoke_llm(llm: LLMRunnable, messages: Sequence[BaseMessage]) -> BaseMessage:
    settings = get_settings()
    try:
        return await llm.ainvoke(list(messages))
    except Exception as exc:
        if not _is_transient(exc):
            raise
        logger.warning("llm_transient_error_retrying", exc_type=type(exc).__name__, error=str(exc))
        await asyncio.sleep(settings.llm_retry_backoff_seconds)
        try:
            return await llm.ainvoke(list(messages))
        except Exception as exc2:
            raise LLMUnavailableError(
                f"ollama unreachable after retry: {type(exc2).__name__}: {exc2}"
            ) from exc2
