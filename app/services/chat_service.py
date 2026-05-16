"""Single entrypoint into the agent graph.

Used by the ``POST /api/chat`` route and by the Chainlit UI (I-09).
Owns the input → graph state mapping, the ``thread_id`` plumbing for
``PostgresSaver`` multi-turn (I-07), the graceful-degradation path on LLM
unavailability, the ``ChatResult`` contract, and the streaming wrapper that
Chainlit consumes (I-09).

The compiled graph (with checkpointer bound) is constructed by the FastAPI
``lifespan`` and injected here. ``ChatService`` does not own checkpointer
lifecycle — that belongs to the app entrypoint or to the CLI debug script.

Streaming model (I-09):

* The UI calls ``astream_ask(...)`` and receives ``StreamChunk`` events.
* A ``reset`` chunk arrives whenever a new ``final_answer``-tagged LLM run
  starts; the UI clears its current buffer. This makes ``route=both``
  consistent: specialist tokens flow first, then Finalize-combined tokens
  replace them.
* ``token`` chunks carry incremental tokens of the current ``final_answer``
  LLM call.
* A final ``result`` chunk carries the full ``ChatResult`` (answer text,
  sources, suggested_action, route metadata) so the UI can render the
  trailing source / action blocks once the stream is done.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from app.graph.llm import FINAL_ANSWER_TAG, LLMUnavailableError
from app.graph.state import UserRole

logger = logging.getLogger(__name__)

_LLM_UNAVAILABLE_MESSAGE = (
    "Сервис AI временно недоступен. Попробуйте повторить запрос через минуту."
)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    suggested_action: dict[str, Any] | None = None
    route: str = "unknown"
    route_reasoning: str | None = None
    thread_id: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamChunk:
    """One event in the UI-facing chat stream.

    Three kinds:

    * ``reset`` — a new final-answer LLM run started; the UI should clear its
      in-progress message buffer.
    * ``token`` — incremental token of the current final-answer LLM call.
    * ``result`` — terminal event with the full ``ChatResult``; emitted once
      after the graph finished.
    """

    kind: Literal["reset", "token", "result"]
    token: str = ""
    result: ChatResult | None = None


class ChatService:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def _initial_state(
        self,
        *,
        question: str,
        user_role: UserRole,
        company_id: int,
        thread_id: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "user_role": user_role,
            "company_id": company_id,
            "thread_id": thread_id,
            "sources": [],
            "errors": [],
            "tool_calls": [],
        }

    def _build_result(
        self,
        final_state: dict[str, Any] | None,
        thread_id: str,
    ) -> ChatResult:
        state = final_state or {}
        return ChatResult(
            answer=str(state.get("final_answer") or "").strip() or _LLM_UNAVAILABLE_MESSAGE,
            sources=list(state.get("sources") or []),
            suggested_action=state.get("suggested_action"),
            route=str(state.get("route") or "unknown"),
            route_reasoning=state.get("route_reasoning"),
            thread_id=thread_id,
            errors=list(state.get("errors") or []),
        )

    def _fallback_result(self, exc: Exception, thread_id: str) -> ChatResult:
        logger.warning("LLM unavailable: %s", exc)
        return ChatResult(
            answer=_LLM_UNAVAILABLE_MESSAGE,
            sources=[],
            suggested_action=None,
            route="error",
            route_reasoning=None,
            thread_id=thread_id,
            errors=[f"llm_unavailable: {exc}"],
        )

    async def ask(
        self,
        *,
        question: str,
        user_role: UserRole,
        company_id: int,
        thread_id: str | None = None,
    ) -> ChatResult:
        thread = thread_id or str(uuid.uuid4())
        initial = self._initial_state(
            question=question, user_role=user_role, company_id=company_id, thread_id=thread
        )
        config = {"configurable": {"thread_id": thread}}
        try:
            final_state = await self._graph.ainvoke(initial, config=config)
        except LLMUnavailableError as exc:
            return self._fallback_result(exc, thread)
        return self._build_result(final_state, thread)

    async def astream_ask(
        self,
        *,
        question: str,
        user_role: UserRole,
        company_id: int,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream final-answer tokens to the UI, then emit a terminal result.

        We listen to ``graph.astream_events`` (v2) and forward chunks of every
        LLM run tagged ``final_answer``. ``on_chat_model_start`` becomes a
        ``reset`` chunk so the UI can clear stale tokens from a prior
        intermediate run (relevant for ``route=both``, where specialists fire
        before Finalize-combined synthesises the real answer).
        """
        thread = thread_id or str(uuid.uuid4())
        initial = self._initial_state(
            question=question, user_role=user_role, company_id=company_id, thread_id=thread
        )
        config = {"configurable": {"thread_id": thread}}

        try:
            async for event in self._graph.astream_events(initial, config=config, version="v2"):
                if FINAL_ANSWER_TAG not in (event.get("tags") or []):
                    continue
                kind = event.get("event")
                if kind == "on_chat_model_start":
                    yield StreamChunk(kind="reset")
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    text = _chunk_text(chunk)
                    if text:
                        yield StreamChunk(kind="token", token=text)
        except LLMUnavailableError as exc:
            yield StreamChunk(kind="result", result=self._fallback_result(exc, thread))
            return

        # After streaming completes the graph has finished. Read the final
        # values from the checkpointer (single source of truth).
        snapshot = await self._graph.aget_state(config)
        yield StreamChunk(kind="result", result=self._build_result(snapshot.values, thread))


def _chunk_text(chunk: Any) -> str:
    """Extract textual content from an LLM stream chunk.

    LangChain emits ``AIMessageChunk`` objects whose ``content`` is usually a
    string; some providers return a list of content blocks. We coerce both to
    a plain string and skip anything that isn't text.
    """
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)
