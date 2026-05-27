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
