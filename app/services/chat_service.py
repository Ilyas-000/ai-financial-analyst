"""Single entrypoint into the agent graph.

Used by the ``POST /api/chat`` route and by the Chainlit UI (I-09).
Owns the input → graph state mapping, the ``thread_id`` plumbing for
``PostgresSaver`` multi-turn (I-07), the graceful-degradation path on LLM
unavailability, the ``ChatResult`` contract, the streaming wrapper that
Chainlit consumes (I-09), and the I/O guardrails (I-11).

The compiled graph (with checkpointer bound) is constructed by the FastAPI
``lifespan`` and injected here. ``ChatService`` does not own checkpointer
lifecycle — that belongs to the app entrypoint or to the CLI debug script.

Guardrails (I-11):

* ``apply_input_guard`` runs before graph entry. Hard blocks (over-length,
  injection hit) short-circuit with a Russian rejection message; PII (cards,
  INN, passport) is masked and the cleaned text is what the graph sees.
* ``apply_output_guard`` runs on the assembled answer after the graph
  finished. Cross-tenant leaks (foreign tenant name or INN in the answer)
  replace the answer with a neutral notice and drop sources + action; cards
  in the answer become ``**** <last4>``; suggested actions without
  ``requires_confirmation=true`` are dropped.

Streaming model (I-09 + I-11 R4):

* Input guard runs synchronously before the graph is entered.
* Tokens from the graph are NOT streamed live (R4, 2026-05-27): we let the
  graph finish, run output guard on the final state, then emit a single
  ``reset`` → ``token`` (full guarded answer) → ``result`` triple. This
  loses live token streaming but guarantees the UI never sees text that
  failed the cross-tenant check.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from app.graph.llm import LLMUnavailableError
from app.graph.state import UserRole
from app.guardrails.input_guard import InputGuardError, apply_input_guard
from app.guardrails.output_guard import apply_output_guard
from app.observability.langfuse_handler import (
    get_langfuse_callback,
    langfuse_trace_attributes,
)

logger = structlog.get_logger(__name__)

_LLM_UNAVAILABLE_MESSAGE = (
    "Сервис AI временно недоступен. Попробуйте повторить запрос через минуту."
)


def synth_user_id(company_id: int, user_role: str) -> str:
    """Build the demo-only synthetic user identity used in audit + logs.

    In production this would be the authenticated user id from the auth
    token; the demo has no real users, so we shape it as
    ``tenant:{company_id}:{role}`` — stable per (tenant, role) but distinct
    across them, which is enough to slice audit and traces.
    """
    return f"tenant:{company_id}:{user_role}"


@contextmanager
def _bound_log_context(
    *,
    thread_id: str,
    user_id: str,
    user_role: str,
    company_id: int,
):
    """Bind per-request log context and clear it on exit.

    Every log line emitted inside the ``with`` block (including from nested
    LangGraph nodes spawned on other asyncio tasks — ``contextvars`` are
    propagated through ``await``) carries these attributes via the
    ``merge_contextvars`` processor configured in ``configure_logging``.
    ``route`` is bound later by ``supervisor_node`` once the routing decision
    is made.
    """
    tokens = structlog.contextvars.bind_contextvars(
        thread_id=thread_id,
        user_id=user_id,
        user_role=user_role,
        company_id=company_id,
    )
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


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

    def _build_config(self, thread_id: str) -> dict[str, Any]:
        """Compose the LangGraph ``RunnableConfig`` for one request.

        ``thread_id`` drives the PostgresSaver checkpointer (I-07). When
        Langfuse is enabled, the shared callback handler is attached here —
        LangChain propagates it down to every nested LLM / tool runnable, so
        nodes stay UI-framework-agnostic.
        """
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        handler = get_langfuse_callback()
        if handler is not None:
            config["callbacks"] = [handler]
        return config

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
            "user_id": synth_user_id(company_id, user_role),
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

    async def _guard_result(
        self,
        *,
        result: ChatResult,
        user_id: str,
        user_company_id: int,
        user_role: UserRole,
        thread_id: str,
    ) -> ChatResult:
        """Pass the assembled answer through the output guard."""
        guarded = await apply_output_guard(
            answer=result.answer,
            sources=result.sources,
            suggested_action=result.suggested_action,
            user_id=user_id,
            user_company_id=user_company_id,
            user_role=user_role,
            thread_id=thread_id,
        )
        errors = list(result.errors)
        if guarded.rejected:
            errors.append(f"output_blocked: {guarded.rejection_reason}")
        if guarded.dropped_action:
            errors.append("output_dropped_action_without_confirmation")
        return ChatResult(
            answer=guarded.answer,
            sources=guarded.sources,
            suggested_action=guarded.suggested_action,
            route="blocked" if guarded.rejected else result.route,
            route_reasoning=result.route_reasoning,
            thread_id=thread_id,
            errors=errors,
        )

    def _blocked_input_result(self, exc: InputGuardError, thread_id: str) -> ChatResult:
        logger.warning("input_blocked", reason=exc.reason, detail=exc.detail)
        return ChatResult(
            answer=str(exc),
            sources=[],
            suggested_action=None,
            route="blocked",
            route_reasoning=None,
            thread_id=thread_id,
            errors=[f"input_blocked: {exc.reason}"],
        )

    def _fallback_result(self, exc: Exception, thread_id: str) -> ChatResult:
        logger.warning("llm_unavailable", error=str(exc))
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
        user_id = synth_user_id(company_id, user_role)
        with _bound_log_context(
            thread_id=thread,
            user_id=user_id,
            user_role=user_role,
            company_id=company_id,
        ):
            try:
                guard_in = apply_input_guard(question)
            except InputGuardError as exc:
                return self._blocked_input_result(exc, thread)
            initial = self._initial_state(
                question=guard_in.question,
                user_role=user_role,
                company_id=company_id,
                thread_id=thread,
            )
            config = self._build_config(thread)
            try:
                with langfuse_trace_attributes(
                    thread_id=thread, user_role=user_role, company_id=company_id
                ):
                    final_state = await self._graph.ainvoke(initial, config=config)
            except LLMUnavailableError as exc:
                return self._fallback_result(exc, thread)
            raw = self._build_result(final_state, thread)
            return await self._guard_result(
                result=raw,
                user_id=user_id,
                user_company_id=company_id,
                user_role=user_role,
                thread_id=thread,
            )

    async def astream_ask(
        self,
        *,
        question: str,
        user_role: UserRole,
        company_id: int,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream-friendly wrapper around ``ask`` (I-11 R4: buffered output).

        Per R4 we no longer surface live LLM tokens to the UI — the output
        guard needs the full answer to detect cross-tenant leaks before the
        user sees anything. We drive the graph to completion, run the guard,
        then emit a single ``reset`` → ``token`` (full guarded answer) →
        ``result`` triple so the existing Chainlit handler contract is
        preserved unchanged.
        """
        thread = thread_id or str(uuid.uuid4())
        user_id = synth_user_id(company_id, user_role)
        with _bound_log_context(
            thread_id=thread,
            user_id=user_id,
            user_role=user_role,
            company_id=company_id,
        ):
            try:
                guard_in = apply_input_guard(question)
            except InputGuardError as exc:
                blocked = self._blocked_input_result(exc, thread)
                yield StreamChunk(kind="reset")
                yield StreamChunk(kind="token", token=blocked.answer)
                yield StreamChunk(kind="result", result=blocked)
                return

            initial = self._initial_state(
                question=guard_in.question,
                user_role=user_role,
                company_id=company_id,
                thread_id=thread,
            )
            config = self._build_config(thread)

            with langfuse_trace_attributes(
                thread_id=thread, user_role=user_role, company_id=company_id
            ):
                try:
                    final_state = await self._graph.ainvoke(initial, config=config)
                except LLMUnavailableError as exc:
                    fallback = self._fallback_result(exc, thread)
                    yield StreamChunk(kind="reset")
                    yield StreamChunk(kind="token", token=fallback.answer)
                    yield StreamChunk(kind="result", result=fallback)
                    return

                raw = self._build_result(final_state, thread)
                guarded = await self._guard_result(
                    result=raw,
                    user_id=user_id,
                    user_company_id=company_id,
                    user_role=user_role,
                    thread_id=thread,
                )

            yield StreamChunk(kind="reset")
            yield StreamChunk(kind="token", token=guarded.answer)
            yield StreamChunk(kind="result", result=guarded)
