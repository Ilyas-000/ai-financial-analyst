"""Integration test for the I-08 ``route="both"`` fan-out + join.

We don't need a live LLM or live infrastructure here — we patch the supervisor,
the two specialist wrappers, and the finalize writer-LLM, then drive the
compiled parent graph end-to-end. The point is to assert that:

* the supervisor's ``both`` route makes the conditional edge fan out to
  ``sql_analyst`` AND ``docs_researcher`` (both wrappers get called exactly
  once with the same question);
* the ``finalize`` join receives both ``sql_result`` and ``docs_result`` in
  the same invocation and produces a single combined final answer;
* both branches' sources are merged in the final state.

The async ``invoke_llm`` is also patched (the condense node's "first turn"
path skips it, but supervisor and finalize both call it).
"""

from unittest.mock import AsyncMock, patch

from app.graph.build import build_agent_graph


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


async def test_route_both_fans_out_and_joins_at_finalize():
    supervisor_payload = (
        '{"next": "both", "reasoning": "fact vs regulation comparison",'
        ' "suggest_action_kind": null}'
    )
    combined_answer = (
        "Расходы по корпкартам за апрель — 612 340,55 ₽, лимит — 500 000 ₽ [1], превышение."
    )

    sql_calls = 0
    docs_calls = 0
    seen_questions: list[str] = []

    async def fake_sql_node(state):
        nonlocal sql_calls
        sql_calls += 1
        seen_questions.append(("sql", state["question"]))
        return {
            "sql_result": {
                "summary": "За апрель потратили 612 340,55 ₽.",
                "sql": "SELECT SUM(t.amount) FROM transactions t WHERE t.company_id = 1",
                "rows_returned": 1,
                "attempts": 1,
                "audit": {"action": "sql_executed"},
            }
        }

    async def fake_docs_node(state):
        nonlocal docs_calls
        docs_calls += 1
        seen_questions.append(("docs", state["question"]))
        return {
            "docs_result": {
                "summary": "Месячный лимит — 500 000 ₽ [1].",
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
        }

    # Each LLM caller (supervisor, finalize) is patched in its own module so
    # the writer-LLM and the supervisor-LLM stay independent.
    async def fake_supervisor_invoke(_llm, _messages):
        return _StubResponse(supervisor_payload)

    async def fake_finalize_invoke(_llm, _messages):
        return _StubResponse(combined_answer)

    with (
        patch("app.graph.build.sql_analyst_node", new=fake_sql_node),
        patch("app.graph.build.docs_researcher_node", new=fake_docs_node),
        patch("app.graph.supervisor.invoke_llm", new=AsyncMock(side_effect=fake_supervisor_invoke)),
        patch("app.graph.finalize.invoke_llm", new=AsyncMock(side_effect=fake_finalize_invoke)),
    ):
        graph = build_agent_graph(checkpointer=None)
        final_state = await graph.ainvoke(
            {
                "question": (
                    "Сравни наши расходы по корпкартам за апрель 2025 с месячным лимитом."
                ),
                "user_role": "finance_manager",
                "company_id": 1,
                "thread_id": "t-i08-1",
                "sources": [],
                "errors": [],
                "tool_calls": [],
            }
        )

    # Both specialists ran, each exactly once.
    assert sql_calls == 1
    assert docs_calls == 1

    # The same condensed question reached both branches.
    questions_seen = {tag: q for tag, q in seen_questions}
    assert questions_seen["sql"] == questions_seen["docs"]
    assert "корпкартам" in questions_seen["sql"]

    # Finalize produced the combined answer and merged sources.
    assert final_state["final_answer"] == combined_answer
    assert final_state["route"] == "both"
    src_types = [s["type"] for s in final_state["sources"]]
    assert sorted(src_types) == ["doc", "sql"]
