"""Unit tests for ``build_trace_steps`` (TD-01 post-hoc agent trace).

Pure, chainlit-free: we feed a ``ChatResult`` and assert the reconstructed
step list. This pins the UI's reasoning-trace contract without importing the
Chainlit app (whose import chain is heavy / unrelated to this logic).
"""

from app.services.chat_service import ChatResult
from app.services.trace_view import build_trace_steps


def test_sql_route_yields_supervisor_then_sql_step():
    result = ChatResult(
        answer="Итого 100 ₽.",
        sources=[
            {
                "type": "sql",
                "sql": "SELECT sum(amount) FROM transactions WHERE company_id = 1",
                "rows_returned": 1,
                "attempts": 2,
            }
        ],
        route="sql_analyst",
        route_reasoning="structured spend question",
    )

    steps = build_trace_steps(result)

    assert [s.name for s in steps] == ["Supervisor", "SQL Analyst"]
    supervisor = steps[0]
    assert supervisor.step_type == "llm"
    assert "SQL-аналитик" in supervisor.content
    assert "structured spend question" in supervisor.content
    sql_step = steps[1]
    assert sql_step.step_type == "tool"
    assert "попыток генерации SQL: 2" in sql_step.content
    assert "```sql" in sql_step.content
    assert "SELECT sum(amount)" in sql_step.content


def test_docs_route_aggregates_into_single_retrieval_step():
    result = ChatResult(
        answer="Согласно регламенту…",
        sources=[
            {"type": "doc", "doc_title": "Travel policy", "section": "3.1", "score": 0.912},
            {"type": "doc", "doc_id": "proc-002"},
        ],
        route="docs_researcher",
        route_reasoning="policy lookup",
    )

    steps = build_trace_steps(result)

    assert [s.name for s in steps] == ["Supervisor", "Docs Researcher"]
    docs = steps[1]
    assert docs.step_type == "retrieval"
    assert "найдено фрагментов: 2" in docs.content
    assert "[1] Travel policy" in docs.content
    assert "§ 3.1" in docs.content
    assert "score=0.912" in docs.content
    assert "[2] proc-002" in docs.content


def test_both_route_shows_sql_then_docs():
    result = ChatResult(
        answer="…",
        sources=[
            {"type": "sql", "sql": "SELECT 1", "rows_returned": 1},
            {"type": "doc", "doc_title": "Регламент"},
        ],
        route="both",
        route_reasoning="data + rules",
    )

    steps = build_trace_steps(result)

    assert [s.name for s in steps] == ["Supervisor", "SQL Analyst", "Docs Researcher"]


def test_direct_answer_has_only_supervisor_step():
    result = ChatResult(answer="Привет!", sources=[], route="direct_answer")

    steps = build_trace_steps(result)

    assert len(steps) == 1
    assert steps[0].name == "Supervisor"
    assert "прямой ответ" in steps[0].content.lower()


def test_errors_append_guardrail_step():
    result = ChatResult(
        answer="Заблокировано.",
        sources=[],
        route="blocked",
        errors=["output_blocked: cross_tenant_leak"],
    )

    steps = build_trace_steps(result)

    assert steps[-1].name == "Guardrails / ошибки"
    assert "cross_tenant_leak" in steps[-1].content
    assert "guardrail" in steps[0].content.lower()


def test_unknown_route_falls_back_to_raw_label():
    result = ChatResult(answer="x", sources=[], route="weird_route")

    steps = build_trace_steps(result)

    assert "weird_route" in steps[0].content
