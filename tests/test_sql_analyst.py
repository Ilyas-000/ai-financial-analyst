"""Regression tests for the SQL Analyst subgraph schema-confusion fix.

Background: the LLM kept generating ``... FROM tariffs t WHERE t.period = ...``
even though ``period`` only exists on ``limits``. The ReAct loop ran out of
attempts because each retry saw only the raw asyncpg first line — no schema
context, no pointer to the table that actually owns the column.

These tests pin down the three pieces of the fix:
  * ``schema_introspect.format_table_for_role`` / ``tables_with_column``
  * ``sql_analyst._build_schema_hint`` and the ``_format_retry_block`` wrapper
  * an end-to-end ReAct flow where the second prompt sees the enriched hint.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.graph import sql_analyst
from app.graph.sql_analyst import (
    _build_schema_hint,
    _extract_failed_column,
    _extract_failed_relation,
    _format_retry_block,
    _tables_in_sql,
    build_sql_analyst_subgraph,
)
from app.tools.schema_introspect import format_table_for_role, tables_with_column
from app.tools.sql_executor import SQLExecutionError
from app.tools.sql_guard import GuardResult

_GENERATE_PROMPT = Path("prompts/sql_analyst_generate.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# schema_introspect helpers
# ---------------------------------------------------------------------------


def test_tables_with_column_finds_period_on_limits_only():
    owners = tables_with_column("period", "finance_manager")
    assert owners == ["limits"]


def test_tables_with_column_returns_empty_for_unknown():
    assert tables_with_column("definitely_not_a_column", "finance_manager") == []


def test_tables_with_column_unknown_role():
    assert tables_with_column("period", "intern") == []


def test_format_table_for_role_includes_columns_and_comment():
    block = format_table_for_role("limits", "finance_manager")
    assert block is not None
    assert "-- limits:" in block
    assert "CREATE TABLE limits" in block
    assert "period VARCHAR(20) NOT NULL" in block
    assert "subject_type VARCHAR(20) NOT NULL" in block


def test_format_table_for_role_hides_sensitive_for_non_privileged():
    # accountant role can read payout_recipients but not the sensitive columns.
    block = format_table_for_role("payout_recipients", "accountant")
    assert block is not None
    assert "inn" not in block
    assert "account" not in block


def test_format_table_for_role_returns_none_when_not_visible():
    # finance_manager has no payout_recipients access.
    assert format_table_for_role("payout_recipients", "finance_manager") is None


# ---------------------------------------------------------------------------
# error parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        ('UndefinedColumnError: column "t.period" does not exist', "period"),
        ("UndefinedColumnError: column t.period does not exist", "period"),
        ('column "period" does not exist', "period"),
        ("column period does not exist", "period"),
    ],
)
def test_extract_failed_column(error, expected):
    assert _extract_failed_column(error) == expected


def test_extract_failed_column_returns_none_for_unrelated_error():
    assert _extract_failed_column("syntax error at or near 'SELECT'") is None


def test_extract_failed_relation():
    assert _extract_failed_relation('relation "ttariffs" does not exist') == "ttariffs"


def test_tables_in_sql_extracts_all_referenced_tables():
    sql = (
        "SELECT t.name, r.fee_type FROM tariffs t "
        "JOIN tariff_rules r ON r.tariff_id = t.id "
        "WHERE t.company_id = 1"
    )
    assert _tables_in_sql(sql) == ["tariff_rules", "tariffs"]


def test_tables_in_sql_handles_parse_error():
    assert _tables_in_sql("not a sql statement at all $$$") == []


# ---------------------------------------------------------------------------
# schema hint / retry block
# ---------------------------------------------------------------------------


_TARIFFS_PERIOD_SQL = (
    "SELECT t.id, t.name, t.period FROM tariffs t WHERE t.company_id = 1 LIMIT 100"
)
_TARIFFS_PERIOD_ERROR = 'UndefinedColumnError: column "t.period" does not exist'


def test_build_schema_hint_points_to_owner_table_for_undefined_column():
    hint = _build_schema_hint(_TARIFFS_PERIOD_SQL, _TARIFFS_PERIOD_ERROR, "finance_manager")

    assert 'column "period" does not exist on the table you used' in hint
    # CREATE TABLE for the table the LLM actually used must be re-emitted.
    assert "CREATE TABLE tariffs" in hint
    assert "effective_from" in hint
    assert "effective_to" in hint
    # And a pointer to where the column does live.
    assert 'Column "period" exists on: limits.' in hint


def test_build_schema_hint_empty_when_error_is_not_schema_related():
    hint = _build_schema_hint(
        "SELECT 1 FROM tariffs t WHERE t.company_id = 1",
        "deadlock detected",
        "finance_manager",
    )
    assert hint == ""


def test_build_schema_hint_handles_unknown_column():
    sql = "SELECT t.frobnicate FROM tariffs t WHERE t.company_id = 1"
    error = 'column "t.frobnicate" does not exist'
    hint = _build_schema_hint(sql, error, "finance_manager")
    assert 'No allowed table has column "frobnicate".' in hint
    assert "CREATE TABLE tariffs" in hint


def test_format_retry_block_wraps_hint_with_previous_sql_and_error():
    block = _format_retry_block(_TARIFFS_PERIOD_SQL, _TARIFFS_PERIOD_ERROR, "finance_manager")
    assert "Previous attempt failed." in block
    assert _TARIFFS_PERIOD_SQL in block
    assert _TARIFFS_PERIOD_ERROR in block
    assert 'Column "period" exists on: limits.' in block
    assert "Fix the SQL while keeping the original intent." in block


def test_format_retry_block_no_op_on_first_attempt():
    assert _format_retry_block(None, None, "finance_manager") == ""


# ---------------------------------------------------------------------------
# prompt regression — make sure the new conventions and few-shots are present
# ---------------------------------------------------------------------------


def test_prompt_documents_tariffs_vs_limits_pitfall():
    assert "`tariffs` has NO `period` column" in _GENERATE_PROMPT
    assert "`limits.period`" in _GENERATE_PROMPT
    assert "effective_from" in _GENERATE_PROMPT


def test_prompt_has_tariff_validity_window_few_shot():
    assert "Какой период действия у нашего активного тарифа?" in _GENERATE_PROMPT
    assert "t.effective_from, t.effective_to" in _GENERATE_PROMPT


def test_prompt_has_limits_by_category_few_shot():
    assert "Какие лимиты по командировкам?" in _GENERATE_PROMPT
    assert "FROM limits l" in _GENERATE_PROMPT
    assert "l.subject_type = 'category'" in _GENERATE_PROMPT
    assert "l.subject_id LIKE 'travel%'" in _GENERATE_PROMPT


# ---------------------------------------------------------------------------
# end-to-end ReAct: bad SQL → enriched retry → good SQL → interpret
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_react_loop_recovers_after_undefined_column_with_schema_hint():
    """Two-turn ReAct: first SQL hits ``UndefinedColumn`` on tariffs.period,
    the second SQL the LLM produces lands on a passing execute_guarded.

    What matters here is that on the *second* generate call, the user prompt
    the LLM sees contains the enriched schema hint — without it the LLM
    would re-emit the same broken SQL until ``sql_max_attempts`` is hit
    (the original bug)."""

    bad_sql = "SELECT t.id, t.period FROM tariffs t WHERE t.company_id = 1 LIMIT 100"
    good_sql = (
        "SELECT t.id, t.name, t.effective_from, t.effective_to "
        "FROM tariffs t WHERE t.company_id = 1 AND t.status = 'active' LIMIT 100"
    )

    captured_user_prompts: list[str] = []
    llm_responses = iter(
        [
            _StubResponse(bad_sql),
            _StubResponse(good_sql),
            _StubResponse("Тариф действует с …"),
        ]
    )

    async def fake_ainvoke(messages):
        # messages = [SystemMessage, HumanMessage] — capture the user one.
        captured_user_prompts.append(messages[-1].content)
        return next(llm_responses)

    class _FakeLLM:
        def with_config(self, _config):
            return self

        async def ainvoke(self, messages):
            return await fake_ainvoke(messages)

    good_guard = GuardResult(
        sql=good_sql,
        tables=frozenset({"tariffs"}),
        audit_action="sql_executed",
        audit_severity="info",
        rewrote_company_id=False,
        rewritten_from_values=(),
        injected_company_id_filters=0,
        enforced_limit=100,
    )

    async def fake_execute_guarded(sql, *, user_id, user_role, company_id, thread_id):
        if "period" in sql:
            raise SQLExecutionError(
                'UndefinedColumnError: column "t.period" does not exist',
                guard_result=good_guard,  # value irrelevant on error
                original=None,
            )
        from app.tools.sql_executor import ExecutionResult

        return ExecutionResult(
            guard=good_guard,
            rows=[
                {
                    "id": 1,
                    "name": "Standard",
                    "effective_from": "2025-01-01",
                    "effective_to": None,
                }
            ],
            rows_returned=1,
            elapsed_ms=3,
        )

    with (
        patch.object(sql_analyst, "make_llm", return_value=_FakeLLM()),
        patch.object(sql_analyst, "execute_guarded", AsyncMock(side_effect=fake_execute_guarded)),
    ):
        graph = build_sql_analyst_subgraph()
        out = await graph.ainvoke(
            {
                "question": "Какой период действия у нашего активного тарифа?",
                "user_id": "tenant:1:finance_manager",
                "user_role": "finance_manager",
                "company_id": 1,
                "thread_id": "t-1",
            }
        )

    # Two generate calls (bad, good) + one interpret call.
    assert len(captured_user_prompts) == 3
    first_prompt, second_prompt, interpret_prompt = captured_user_prompts

    # First generate: no retry block yet.
    assert "Previous attempt failed." not in first_prompt

    # Second generate: the enriched hint is what makes the fix work.
    assert "Previous attempt failed." in second_prompt
    assert bad_sql in second_prompt
    assert 'column "period" does not exist on the table you used' in second_prompt
    assert "CREATE TABLE tariffs" in second_prompt
    assert 'Column "period" exists on: limits.' in second_prompt

    # Interpret got the rows from the second attempt.
    assert "effective_from" in interpret_prompt

    assert out["attempts"] == 2
    assert out["error"] is None
    assert out["sql"] == good_sql
    assert out["summary"].startswith("Тариф действует")
