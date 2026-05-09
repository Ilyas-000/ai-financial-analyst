"""Single entrypoint into the agent graph.

Used by the ``POST /api/chat`` route today and by the Chainlit UI from I-09.
Owns the input → graph state mapping, the graceful-degradation path on LLM
unavailability, and the ``ChatResult`` contract.
"""

import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.graph.build import get_agent_graph
from app.graph.llm import LLMUnavailableError
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


class ChatService:
    def __init__(self, graph: Any | None = None) -> None:
        self._graph = graph or get_agent_graph()

    async def ask(
        self,
        *,
        question: str,
        user_role: UserRole,
        company_id: int,
        thread_id: str | None = None,
    ) -> ChatResult:
        thread = thread_id or str(uuid.uuid4())
        initial: dict[str, Any] = {
            "question": question,
            "user_role": user_role,
            "company_id": company_id,
            "thread_id": thread,
            "sources": [],
            "errors": [],
            "tool_calls": [],
        }
        try:
            final_state = await self._graph.ainvoke(initial)
        except LLMUnavailableError as exc:
            logger.warning("LLM unavailable: %s", exc)
            return ChatResult(
                answer=_LLM_UNAVAILABLE_MESSAGE,
                sources=[],
                suggested_action=None,
                route="error",
                route_reasoning=None,
                thread_id=thread,
                errors=[f"llm_unavailable: {exc}"],
            )

        return ChatResult(
            answer=str(final_state.get("final_answer") or "").strip() or _LLM_UNAVAILABLE_MESSAGE,
            sources=list(final_state.get("sources") or []),
            suggested_action=final_state.get("suggested_action"),
            route=str(final_state.get("route") or "unknown"),
            route_reasoning=final_state.get("route_reasoning"),
            thread_id=thread,
            errors=list(final_state.get("errors") or []),
        )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService()
