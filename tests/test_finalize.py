"""Unit tests for the Finalize aggregator.

Covers:

* I-06 single-source paths (sql_only / docs_only / fallback).
* I-06 suggest_action_kind branches export_report / open_ticket.
* TD-11 deterministic entity extraction for prepare_act (counterparty + payout
  ids) and highlight_discrepancy (expected/actual figures from the summaries);
  both still skip gracefully when the required entities are absent.
* I-07 ``AIMessage(final_answer)`` emission so PostgresSaver persists the
  assistant turn.
* I-08 combined synthesis path (``route="both"``): writer-LLM is called with
  both summaries and its output becomes ``final_answer``. Empty LLM output
  falls back to a deterministic concatenation.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.graph.finalize import finalize_node


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def _base_state() -> dict:
    return {
        "question": "Сколько потратили на командировки за Q1?",
        "user_role": "finance_manager",
        "company_id": 1,
        "thread_id": "t-1",
        "route": "sql_analyst",
        "route_reasoning": "needs aggregation",
        "suggest_action_kind": None,
    }


async def test_sql_only_summary_and_sources():
    state = _base_state()
    state["sql_result"] = {
        "summary": "За Q1 потратили 1 234 567,89 ₽.",
        "sql": "SELECT SUM(t.amount) FROM transactions t WHERE t.company_id = 1",
        "rows": [{"sum": 1234567.89}],
        "rows_returned": 1,
        "audit": {"action": "sql_executed"},
        "attempts": 1,
        "error": None,
    }

    out = await finalize_node(state)

    assert out["final_answer"].startswith("За Q1")
    assert len(out["sources"]) == 1
    src = out["sources"][0]
    assert src["type"] == "sql"
    assert src["rows_returned"] == 1
    assert src["attempts"] == 1
    assert "company_id = 1" in src["sql"]
    assert out["suggested_action"] is None


async def test_docs_only_summary_and_sources():
    state = _base_state()
    state["route"] = "docs_researcher"
    state["docs_result"] = {
        "summary": "Лимит топ-менеджеров: 500 000 ₽ в месяц [1].",
        "sources": [
            {
                "n": 1,
                "doc_id": "policy_cards",
                "doc_title": "Политика корпкарт",
                "section": "3.2",
                "tenant_scope": "tenant:*",
                "score": 0.97,
            }
        ],
    }

    out = await finalize_node(state)

    assert "Лимит" in out["final_answer"]
    assert len(out["sources"]) == 1
    src = out["sources"][0]
    assert src["type"] == "doc"
    assert src["doc_id"] == "policy_cards"
    assert src["score"] == 0.97
    assert out["suggested_action"] is None


async def test_fallback_when_no_specialist_summary():
    state = _base_state()
    state["sql_result"] = {"summary": "", "sql": "", "rows_returned": 0}

    out = await finalize_node(state)

    assert "Не удалось" in out["final_answer"]
    assert out["sources"] == []
    assert out["suggested_action"] is None


async def test_export_report_action_built_from_hint():
    state = _base_state()
    state["suggest_action_kind"] = "export_report"
    state["sql_result"] = {
        "summary": "Готов сводный отчёт.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }

    out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "export_report"
    assert action["requires_confirmation"] is True
    assert action["payload"]["currency"] == "RUB"
    assert action["payload"]["format"] == "csv"


async def test_open_ticket_action_built_from_hint():
    state = _base_state()
    state["question"] = "У нас расхождение по комиссии за payout, эскалируйте."
    state["suggest_action_kind"] = "open_ticket"
    state["sql_result"] = {"summary": "Видим разницу 1 200 ₽.", "sql": "SELECT 1"}

    out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "open_ticket"
    assert action["payload"]["severity"] == "medium"
    assert "расхождение" in action["payload"]["summary"]


async def test_prepare_act_skipped_when_entities_missing():
    # No counterparty in the question and no payout ids in the rows → skip.
    state = _base_state()
    state["suggest_action_kind"] = "prepare_act"
    state["sql_result"] = {"summary": "Сформирован список выплат.", "sql": "SELECT 1"}

    out = await finalize_node(state)

    assert out["suggested_action"] is None


async def test_prepare_act_built_from_counterparty_and_payout_ids():
    state = _base_state()
    state["question"] = "По подрядчику ООО Ромашка нужны закрывающие за март"
    state["suggest_action_kind"] = "prepare_act"
    state["sql_result"] = {
        "summary": "Найдено 2 выплаты подрядчику.",
        "sql": "SELECT id FROM payouts WHERE company_id = 1",
        "rows": [{"payout_id": 101}, {"payout_id": 102}],
        "rows_returned": 2,
    }

    out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "prepare_act"
    assert action["requires_confirmation"] is True
    assert action["payload"]["counterparty_name"] == "ООО Ромашка"
    assert action["payload"]["payout_ids"] == [101, 102]


async def test_prepare_act_uses_id_column_when_no_payout_id():
    state = _base_state()
    state["question"] = "Закрывающие для ИП Иванов за период"
    state["suggest_action_kind"] = "prepare_act"
    state["sql_result"] = {
        "summary": "Выплаты найдены.",
        "sql": "SELECT id FROM payouts",
        "rows": [{"id": 7}, {"id": 9}],
        "rows_returned": 2,
    }

    out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["payload"]["counterparty_name"] == "ИП Иванов"
    assert action["payload"]["payout_ids"] == [7, 9]


async def test_highlight_discrepancy_skipped_when_no_figures():
    state = _base_state()
    state["suggest_action_kind"] = "highlight_discrepancy"
    state["sql_result"] = {"summary": "Нашли расхождение.", "sql": "SELECT 1"}

    out = await finalize_node(state)

    assert out["suggested_action"] is None


async def test_highlight_discrepancy_built_from_both_summaries():
    """expected = docs figure, actual = SQL figure — both anchored to ₽, so a
    bare year earlier in the sentence ("апрель 2025") is not mistaken for it."""
    state = _base_state()
    state["route"] = "both"
    state["suggest_action_kind"] = "highlight_discrepancy"
    state["sql_result"] = {
        "summary": "За апрель 2025 расходы по корпкартам — 612 340,55 ₽.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }
    state["docs_result"] = {
        "summary": "Месячный лимит на корпкарту — 500 000 ₽ [1].",
        "sources": [{"n": 1, "doc_id": "policy_cards", "score": 0.94}],
    }

    with patch(
        "app.graph.finalize.invoke_llm",
        return_value=_StubResponse("Расходы превышают лимит."),
    ):
        out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "highlight_discrepancy"
    assert action["payload"]["actual"] == "612 340,55"
    assert action["payload"]["expected"] == "500 000"


async def test_messages_emitted_for_persistence():
    state = _base_state()
    state["sql_result"] = {
        "summary": "За Q1 потратили 1 234 567,89 ₽.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }

    out = await finalize_node(state)

    messages = out["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == out["final_answer"]


async def test_action_built_from_condensed_question():
    """``open_ticket`` topic must use the standalone question, not the raw follow-up."""
    state = _base_state()
    state["question"] = "А по этому?"
    state["condensed_question"] = "Эскалируйте расхождение по комиссии за payout от 2025-04-12."
    state["suggest_action_kind"] = "open_ticket"
    state["sql_result"] = {"summary": "Видим разницу 1 200 ₽.", "sql": "SELECT 1"}

    out = await finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert "расхождение" in action["payload"]["summary"]
    assert "payout" in action["payload"]["topic"]


async def test_both_route_synthesises_with_writer_llm():
    """When both branches produced a summary, Finalize calls the writer LLM
    with the question + both summaries and uses its output as the final answer.
    Sources from both branches are merged. The synthesis prompt receives the
    condensed (standalone) question, not the raw follow-up."""
    state = _base_state()
    state["route"] = "both"
    state["question"] = "А по этому?"
    state["condensed_question"] = (
        "Сравни наши расходы по корпкартам за апрель 2025 с месячным лимитом из регламента."
    )
    state["sql_result"] = {
        "summary": "За апрель 2025 расходы по корпкартам — 612 340,55 ₽.",
        "sql": "SELECT SUM(t.amount) FROM transactions t WHERE t.company_id = 1",
        "rows_returned": 1,
        "attempts": 1,
        "audit": {"action": "sql_executed"},
    }
    state["docs_result"] = {
        "summary": "Месячный лимит на корпкарту топ-менеджера — 500 000 ₽ [1].",
        "sources": [
            {
                "n": 1,
                "doc_id": "policy_cards",
                "doc_title": "Политика корпкарт",
                "section": "3.2",
                "tenant_scope": "tenant:*",
                "score": 0.94,
            }
        ],
    }

    synthesized = (
        "Расходы по корпкартам за апрель 2025 составили 612 340,55 ₽, "
        "что превышает месячный лимит 500 000 ₽ [1]."
    )

    seen_payload: dict[str, object] = {}

    async def fake_invoke(_llm, messages):
        # capture the user prompt so we can verify the synthesis sees both summaries
        seen_payload["user"] = messages[-1].content
        return _StubResponse(synthesized)

    with patch("app.graph.finalize.invoke_llm", side_effect=fake_invoke):
        out = await finalize_node(state)

    assert out["final_answer"] == synthesized
    assert "612 340,55" in seen_payload["user"]
    assert "500 000 ₽" in seen_payload["user"]
    # condensed question is what reaches the writer, not the raw follow-up
    assert "корпкартам за апрель 2025" in seen_payload["user"]
    assert "А по этому?" not in seen_payload["user"]

    # merged sources: one SQL + one doc
    types = [src["type"] for src in out["sources"]]
    assert types == ["sql", "doc"]


async def test_both_route_falls_back_to_concat_on_empty_synthesis():
    """If the writer LLM returns an empty string, Finalize falls back to a
    deterministic concatenation so the user still sees both summaries."""
    state = _base_state()
    state["route"] = "both"
    state["sql_result"] = {
        "summary": "За апрель потратили 612 340,55 ₽.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }
    state["docs_result"] = {
        "summary": "Лимит — 500 000 ₽ [1].",
        "sources": [{"n": 1, "doc_id": "policy_cards", "score": 0.9}],
    }

    with patch(
        "app.graph.finalize.invoke_llm",
        return_value=_StubResponse("   \n"),
    ):
        out = await finalize_node(state)

    assert "612 340,55" in out["final_answer"]
    assert "500 000" in out["final_answer"]
    assert len(out["sources"]) == 2


async def test_both_route_skips_llm_when_one_summary_empty():
    """If only one branch produced a summary, fall back to the single-source
    path (no writer-LLM call) — same behaviour as routes ``sql_analyst`` /
    ``docs_researcher``. This handles the case where a specialist failed."""
    state = _base_state()
    state["route"] = "both"
    state["sql_result"] = {
        "summary": "За апрель потратили 612 340,55 ₽.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }
    state["docs_result"] = {"summary": "", "sources": []}

    with patch("app.graph.finalize.invoke_llm") as mocked_invoke:
        out = await finalize_node(state)

    mocked_invoke.assert_not_called()
    assert out["final_answer"] == "За апрель потратили 612 340,55 ₽."
    # docs branch contributed no sources
    assert len(out["sources"]) == 1
    assert out["sources"][0]["type"] == "sql"
