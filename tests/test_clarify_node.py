"""Smoke test for ``clarify_node`` template formatting.

Hotfix coverage: ``{reasoning}`` lives in the SYSTEM section of
``prompts/clarify.txt`` but used to be passed only into the USER format call,
which exploded with ``KeyError: 'reasoning'`` whenever Supervisor routed to
clarify. This test exercises the format step with a stubbed LLM so the bug
can't silently come back.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

import app.graph.clarify as clarify_module
from app.graph.clarify import clarify_node

pytestmark = pytest.mark.asyncio


class _StubLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_messages: list[Any] | None = None

    async def ainvoke(self, messages, config=None):
        self.last_messages = list(messages)
        return AIMessage(content=self._reply)


async def test_clarify_node_formats_prompt_without_keyerror(monkeypatch):
    async def _slug(_company_id: int) -> str:
        return "acme"

    stub = _StubLLM("Уточните, пожалуйста, период.")

    async def _invoke(llm, messages):
        return await llm.ainvoke(messages)

    monkeypatch.setattr(clarify_module, "tenant_slug_for", _slug)
    monkeypatch.setattr(clarify_module, "make_llm", lambda role, *, tags=None: stub)
    monkeypatch.setattr(clarify_module, "invoke_llm", _invoke)

    state = {
        "question": "Карта **** 1111 не работает.",
        "user_role": "finance_manager",
        "company_id": 1,
        "route_reasoning": "needs period",
    }

    result = await clarify_node(state)

    assert result["final_answer"] == "Уточните, пожалуйста, период."
    assert result["sources"] == []
    assert result["suggested_action"] is None
    # SYSTEM prompt received the reasoning placeholder — guarantees the bug
    # (KeyError before the LLM is reached) is dead.
    system_msg = stub.last_messages[0]
    assert "needs period" in system_msg.content
    assert "finance_manager" in system_msg.content
    assert "acme" in system_msg.content


async def test_clarify_node_uses_default_reasoning_when_state_missing(monkeypatch):
    async def _slug(_company_id: int) -> str:
        return "acme"

    stub = _StubLLM("ok")

    async def _invoke(llm, messages):
        return await llm.ainvoke(messages)

    monkeypatch.setattr(clarify_module, "tenant_slug_for", _slug)
    monkeypatch.setattr(clarify_module, "make_llm", lambda role, *, tags=None: stub)
    monkeypatch.setattr(clarify_module, "invoke_llm", _invoke)

    state: dict[str, Any] = {
        "question": "что-то",
        "user_role": "finance_manager",
        "company_id": 1,
    }

    await clarify_node(state)

    system_msg = stub.last_messages[0]
    assert "ambiguous question" in system_msg.content


def _patch_clarify_llm(monkeypatch, reply: str = "Уточните период.") -> None:
    async def _slug(_company_id: int) -> str:
        return "acme"

    stub = _StubLLM(reply)

    async def _invoke(llm, messages):
        return await llm.ainvoke(messages)

    monkeypatch.setattr(clarify_module, "tenant_slug_for", _slug)
    monkeypatch.setattr(clarify_module, "make_llm", lambda role, *, tags=None: stub)
    monkeypatch.setattr(clarify_module, "invoke_llm", _invoke)


async def test_clarify_node_preserves_action_intent_as_pre_action(monkeypatch):
    """TD-10: an action hint that arrives with route=clarify must not be dropped.

    Finalize never runs on the clarify branch, so clarify itself emits a draft
    in a ``needs_clarification`` state to keep the user's intent for the turn.
    """
    _patch_clarify_llm(monkeypatch)

    state = {
        "question": "Открой тикет по подозрительной активности на корпкарте",
        "user_role": "cfo",
        "company_id": 1,
        "route_reasoning": "missing period",
        "suggest_action_kind": "open_ticket",
    }

    result = await clarify_node(state)

    action = result["suggested_action"]
    assert action is not None
    assert action["kind"] == "open_ticket"
    assert action["state"] == "needs_clarification"
    assert action["requires_confirmation"] is True
    assert "подозрительной активности" in action["payload"]["question_excerpt"]


async def test_clarify_node_no_action_when_no_hint(monkeypatch):
    _patch_clarify_llm(monkeypatch)

    state = {
        "question": "Сколько мы потратили?",
        "user_role": "finance_manager",
        "company_id": 1,
        "route_reasoning": "no scope",
    }

    result = await clarify_node(state)

    assert result["suggested_action"] is None
