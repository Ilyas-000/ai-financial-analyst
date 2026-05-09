"""Unit tests for the Finalize aggregator.

Covers the I-06 single-source paths plus the suggest_action_kind branches we
actually build (export_report, open_ticket). prepare_act / highlight_discrepancy
are deliberately skipped in I-06 (need entity extraction); we assert that.
"""

from app.graph.finalize import finalize_node


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


def test_sql_only_summary_and_sources():
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

    out = finalize_node(state)

    assert out["final_answer"].startswith("За Q1")
    assert len(out["sources"]) == 1
    src = out["sources"][0]
    assert src["type"] == "sql"
    assert src["rows_returned"] == 1
    assert src["attempts"] == 1
    assert "company_id = 1" in src["sql"]
    assert out["suggested_action"] is None


def test_docs_only_summary_and_sources():
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

    out = finalize_node(state)

    assert "Лимит" in out["final_answer"]
    assert len(out["sources"]) == 1
    src = out["sources"][0]
    assert src["type"] == "doc"
    assert src["doc_id"] == "policy_cards"
    assert src["score"] == 0.97
    assert out["suggested_action"] is None


def test_fallback_when_no_specialist_summary():
    state = _base_state()
    state["sql_result"] = {"summary": "", "sql": "", "rows_returned": 0}

    out = finalize_node(state)

    assert "Не удалось" in out["final_answer"]
    assert out["sources"] == []
    assert out["suggested_action"] is None


def test_export_report_action_built_from_hint():
    state = _base_state()
    state["suggest_action_kind"] = "export_report"
    state["sql_result"] = {
        "summary": "Готов сводный отчёт.",
        "sql": "SELECT 1",
        "rows_returned": 1,
    }

    out = finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "export_report"
    assert action["requires_confirmation"] is True
    assert action["payload"]["currency"] == "RUB"
    assert action["payload"]["format"] == "csv"


def test_open_ticket_action_built_from_hint():
    state = _base_state()
    state["question"] = "У нас расхождение по комиссии за payout, эскалируйте."
    state["suggest_action_kind"] = "open_ticket"
    state["sql_result"] = {"summary": "Видим разницу 1 200 ₽.", "sql": "SELECT 1"}

    out = finalize_node(state)

    action = out["suggested_action"]
    assert action is not None
    assert action["kind"] == "open_ticket"
    assert action["payload"]["severity"] == "medium"
    assert "расхождение" in action["payload"]["summary"]


def test_prepare_act_hint_skipped_until_entity_extraction():
    state = _base_state()
    state["suggest_action_kind"] = "prepare_act"
    state["sql_result"] = {"summary": "Сформирован список выплат.", "sql": "SELECT 1"}

    out = finalize_node(state)

    # Hint preserved on state for tracing, but action not built yet.
    assert out["suggested_action"] is None


def test_highlight_discrepancy_hint_skipped_until_entity_extraction():
    state = _base_state()
    state["suggest_action_kind"] = "highlight_discrepancy"
    state["sql_result"] = {"summary": "Нашли расхождение.", "sql": "SELECT 1"}

    out = finalize_node(state)

    assert out["suggested_action"] is None
