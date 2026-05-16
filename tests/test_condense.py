"""Unit tests for the condense_question node and its helpers.

We test the pure helpers (history formatting, output unwrapping, prior-turn
detection) directly, and exercise the node twice with a stub LLM:

* First turn → no prior assistant message → LLM must NOT be called and the
  raw question is echoed verbatim into ``condensed_question``.
* Follow-up turn → LLM IS called and its output becomes ``condensed_question``.

Live LLM coverage happens in scripts/chat.py during I-07 smoke.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.condense import (
    _format_history,
    _has_prior_assistant_turn,
    _strip_wrappers,
    condense_question_node,
)


def test_strip_wrappers_removes_code_fences():
    assert _strip_wrappers("```\nfoo\n```") == "foo"
    assert _strip_wrappers("```text\nfoo\n```") == "foo"


def test_strip_wrappers_removes_matched_quotes():
    assert _strip_wrappers('"foo bar"') == "foo bar"
    assert _strip_wrappers("'foo bar'") == "foo bar"
    assert _strip_wrappers("«foo bar»") == "foo bar"
    # mismatched should be left alone
    assert _strip_wrappers('"foo bar') == '"foo bar'


def test_has_prior_assistant_turn():
    assert not _has_prior_assistant_turn([])
    assert not _has_prior_assistant_turn([HumanMessage(content="hi")])
    assert _has_prior_assistant_turn(
        [HumanMessage(content="hi"), AIMessage(content="hello")]
    )


def test_format_history_uses_recent_turns_and_labels():
    msgs = [
        HumanMessage(content="Сколько потратили на командировки за Q1?"),
        AIMessage(content="В Q1 потратили 1 234 567 ₽."),
        HumanMessage(content="А по департаменту маркетинга?"),
        AIMessage(content="Маркетинг — 312 000 ₽."),
    ]
    rendered = _format_history(msgs)
    lines = rendered.splitlines()
    assert lines[0].startswith("User: ")
    assert lines[1].startswith("Assistant: ")
    assert "Q1" in rendered
    assert "Маркетинг" in rendered


def test_format_history_truncates_long_messages():
    long_text = "x" * 2000
    rendered = _format_history([HumanMessage(content=long_text)])
    line = rendered.splitlines()[0]
    # length includes "User: " prefix; ellipsis appended after truncation
    assert line.endswith("…")
    assert len(line) < 700


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


async def test_first_turn_skips_llm_and_echoes_question():
    state = {
        "question": "Сколько потратили на командировки за последние 90 дней?",
        "user_role": "finance_manager",
        "company_id": 1,
    }

    with patch("app.graph.condense.invoke_llm") as mocked_invoke:
        out = await condense_question_node(state)

    mocked_invoke.assert_not_called()
    assert out["condensed_question"] == state["question"]
    messages = out["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == state["question"]


async def test_followup_turn_uses_llm_rewrite():
    state = {
        "question": "а за Q2?",
        "user_role": "finance_manager",
        "company_id": 1,
        "messages": [
            HumanMessage(content="Сколько потратили на командировки в Q1 2025?"),
            AIMessage(content="В Q1 2025 потратили 1 234 567 ₽."),
        ],
    }

    rewritten = "Сколько потратили на командировки в Q2 2025?"
    with patch(
        "app.graph.condense.invoke_llm",
        return_value=_StubResponse(rewritten),
    ) as mocked_invoke:
        out = await condense_question_node(state)

    mocked_invoke.assert_called_once()
    assert out["condensed_question"] == rewritten
    # new user turn appended (not the whole history; add_messages reducer concatenates)
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "а за Q2?"


async def test_followup_strips_quotes_and_fences():
    state = {
        "question": "а за Q2?",
        "user_role": "finance_manager",
        "company_id": 1,
        "messages": [
            HumanMessage(content="Сколько потратили на командировки в Q1?"),
            AIMessage(content="1 234 567 ₽."),
        ],
    }

    raw = '```\n"Сколько потратили на командировки в Q2?"\n```'
    with patch("app.graph.condense.invoke_llm", return_value=_StubResponse(raw)):
        out = await condense_question_node(state)

    assert out["condensed_question"] == "Сколько потратили на командировки в Q2?"


async def test_followup_falls_back_when_rewrite_empty():
    state = {
        "question": "а за Q2?",
        "user_role": "finance_manager",
        "company_id": 1,
        "messages": [
            HumanMessage(content="Сколько потратили на командировки в Q1?"),
            AIMessage(content="ответ"),
        ],
    }

    with patch("app.graph.condense.invoke_llm", return_value=_StubResponse("   \n")):
        out = await condense_question_node(state)

    assert out["condensed_question"] == "а за Q2?"


@pytest.mark.parametrize(
    "tail_marker",
    [
        "Standalone:",  # in case LLM ignores rule and adds a label
        "Q:",
    ],
)
def test_strip_wrappers_leaves_labels_alone(tail_marker):
    # We deliberately do NOT strip arbitrary labels — the prompt forbids them,
    # so a label slipping through is a prompt-regression signal, not noise.
    text = f"{tail_marker} foo"
    assert _strip_wrappers(text) == text


async def test_condense_resets_per_turn_state():
    """Stale per-turn fields from the prior checkpoint must be wiped.

    Without the reset, a turn-N route of ``sql_analyst`` would land at Finalize
    with the turn N-1 ``docs_result`` still present in state and surface its
    docs sources alongside the new SQL result.
    """
    state = {
        "question": "А по департаменту маркетинга?",
        "user_role": "finance_manager",
        "company_id": 1,
        # Leftovers from a previous docs_researcher turn:
        "route": "docs_researcher",
        "route_reasoning": "policy lookup",
        "suggest_action_kind": "export_report",
        "docs_result": {"summary": "stale", "sources": [{"n": 1}]},
        "sql_result": None,
        "final_answer": "stale answer",
        "sources": [{"type": "doc", "n": 1}],
        "suggested_action": {"kind": "export_report"},
        "errors": ["old error"],
        "tool_calls": [{"name": "old"}],
        "messages": [
            HumanMessage(content="Сколько потратили на командировки за 90 дней?"),
            AIMessage(content="1 784 225,07 ₽."),
        ],
    }

    rewritten = "Сколько потратили на маркетинг за последние 90 дней?"
    with patch(
        "app.graph.condense.invoke_llm",
        return_value=_StubResponse(rewritten),
    ):
        out = await condense_question_node(state)

    # condensed_question filled
    assert out["condensed_question"] == rewritten
    # every single-turn field reset
    assert out["route"] is None
    assert out["route_reasoning"] is None
    assert out["suggest_action_kind"] is None
    assert out["sql_result"] is None
    assert out["docs_result"] is None
    assert out["final_answer"] is None
    assert out["sources"] == []
    assert out["suggested_action"] is None
    assert out["errors"] == []
    assert out["tool_calls"] == []
    # messages NOT reset — multi-turn log accumulates via add_messages reducer
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == state["question"]
