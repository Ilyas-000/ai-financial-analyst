"""SQL Analyst subgraph: generate → execute (guarded) → interpret, with ReAct retry.

Nodes: ``generate_sql`` (LLM emits one Postgres SELECT; on retry it also sees
the prior SQL + error), ``execute`` (``execute_guarded`` → ``sql_guard`` +
``audit_log``; guard rejections and DB errors both feed the retry loop),
``interpret`` (second LLM call summarises rows in Russian), ``fail`` (terminal
fallback once ``sql_max_attempts`` is exhausted). ``execute`` loops back to
``generate_sql`` while ``attempt < max_attempts``.

State contract:
    in:  ``question``, ``user_role``, ``company_id``, ``thread_id``
    out: ``summary``, ``sql``, ``rows``, ``rows_returned``, ``audit``,
         ``attempts``, ``error``
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypedDict

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from sqlglot import exp

from app.config import get_settings
from app.graph.llm import FINAL_ANSWER_TAG, invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt, strip_code_fences
from app.tools.schema_introspect import (
    format_table_for_role,
    schema_for_role,
    tables_with_column,
)
from app.tools.sql_executor import (
    ExecutionResult,
    SQLExecutionError,
    execute_guarded,
)
from app.tools.sql_guard import GuardRejection

_GENERATE_PROMPT_FILE = "sql_analyst_generate.txt"
_INTERPRET_PROMPT_FILE = "sql_analyst_interpret.txt"

_INTERPRET_ROW_CAP = 50

_FAIL_FALLBACK = (
    "Не удалось получить ответ из базы за {attempts} попыт(ок). Последняя ошибка: {error}"
)
_EMPTY_RESULT_FALLBACK = "По заданным условиям записей не найдено."


class SQLAnalystState(TypedDict, total=False):
    question: str
    user_id: str
    user_role: str
    company_id: int
    thread_id: str | None

    attempt: int
    candidate_sql: str
    last_error: str | None

    sql: str
    rows: list[dict[str, Any]]
    rows_returned: int
    elapsed_ms: int
    audit: dict[str, Any]
    summary: str
    attempts: int
    error: str | None


# Postgres error text. We rely on the *first line* (see ``_short_db_error``),
# which is the message string of UndefinedColumnError / UndefinedTableError —
# always exactly ``column "..." does not exist`` / ``relation "..." does not exist``.
_UNDEFINED_COLUMN_RE = re.compile(r'column\s+"?([^\s"]+)"?\s+does not exist', re.IGNORECASE)
_UNDEFINED_TABLE_RE = re.compile(r'relation\s+"?([^\s"]+)"?\s+does not exist', re.IGNORECASE)


def _extract_failed_column(error: str) -> str | None:
    """Return the bare column name from a Postgres UndefinedColumn error.

    Postgres reports the column as written in the SQL — either ``"col"`` or
    ``"alias.col"``. We strip the alias because the schema hint indexes by
    column name, not by SQL fragment.
    """
    match = _UNDEFINED_COLUMN_RE.search(error)
    if not match:
        return None
    raw = match.group(1)
    return raw.rsplit(".", 1)[-1] if "." in raw else raw


def _extract_failed_relation(error: str) -> str | None:
    match = _UNDEFINED_TABLE_RE.search(error)
    return match.group(1).rsplit(".", 1)[-1] if match else None


def _tables_in_sql(sql: str) -> list[str]:
    # sqlglot raises a small family of subclasses (ParseError, TokenError, …);
    # the retry-hint path is best-effort, so swallow them all and fall back
    # to an empty list rather than crash the second generate call.
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.SqlglotError:
        return []
    if parsed is None:
        return []
    return sorted({(t.name or "").lower() for t in parsed.find_all(exp.Table) if t.name})


def _build_schema_hint(last_sql: str, last_error: str, user_role: str) -> str:
    """Schema slice to append on retry when Postgres rejected a column/table.

    The LLM otherwise sees only the raw asyncpg error and routinely keeps
    re-trying the same wrong table — the structural fix is to re-emit the
    real CREATE TABLE for the tables it actually referenced, plus a pointer
    to where the missing column does live.
    """
    column = _extract_failed_column(last_error)
    relation = _extract_failed_relation(last_error) if column is None else None
    if column is None and relation is None:
        return ""

    touched = _tables_in_sql(last_sql)
    create_blocks = [
        block for name in touched if (block := format_table_for_role(name, user_role)) is not None
    ]
    parts: list[str] = []
    if column is not None:
        owners = tables_with_column(column, user_role)
        parts.append(f'Schema hint — column "{column}" does not exist on the table you used.')
        if create_blocks:
            parts.append("Real schema of the tables in your previous SQL:")
            parts.append("\n\n".join(create_blocks))
        if owners:
            owners_str = ", ".join(owners)
            parts.append(f'Column "{column}" exists on: {owners_str}.')
        else:
            parts.append(f'No allowed table has column "{column}".')
    else:
        parts.append(f'Schema hint — relation "{relation}" is not an allowed table.')

    return "\n".join(parts) + "\n\n"


def _format_retry_block(last_sql: str | None, last_error: str | None, user_role: str) -> str:
    if not last_sql or not last_error:
        return ""
    hint = _build_schema_hint(last_sql, last_error, user_role)
    return (
        "Previous attempt failed.\n"
        "Previous SQL:\n"
        f"{last_sql}\n\n"
        "Error:\n"
        f"{last_error}\n\n"
        f"{hint}"
        "Fix the SQL while keeping the original intent.\n\n"
    )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _render_rows_table(rows: list[dict[str, Any]], cap: int) -> str:
    if not rows:
        return "(нет строк)"
    shown = rows[:cap]
    columns = list(shown[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = [
        "| " + " | ".join(_format_value(row.get(c)) for c in columns) + " |" for row in shown
    ]
    return "\n".join([header, separator, *body_lines])


async def _generate_sql_node(state: SQLAnalystState) -> SQLAnalystState:
    attempt = int(state.get("attempt", 0)) + 1
    system_prompt, user_template = load_two_section_prompt(_GENERATE_PROMPT_FILE)
    schema = schema_for_role(state["user_role"])
    user_prompt = user_template.format(
        schema=schema,
        question=state["question"],
        retry_block=_format_retry_block(
            state.get("candidate_sql"),
            state.get("last_error"),
            state["user_role"],
        ),
    )
    system_filled = system_prompt.format(
        user_role=state["user_role"],
        company_id=state["company_id"],
    )
    llm = make_llm("specialist")
    response = await invoke_llm(llm, [SystemMessage(system_filled), HumanMessage(user_prompt)])
    candidate = strip_code_fences(str(response.content)).rstrip(";").strip()
    return {
        "attempt": attempt,
        "candidate_sql": candidate,
        "attempts": attempt,
        # Reset transient error so the retry block is rebuilt only on real failures.
        "last_error": None,
    }


async def _execute_node(state: SQLAnalystState) -> SQLAnalystState:
    candidate = state.get("candidate_sql", "")
    try:
        result: ExecutionResult = await execute_guarded(
            candidate,
            user_id=state["user_id"],
            user_role=state["user_role"],
            company_id=state["company_id"],
            thread_id=state.get("thread_id"),
        )
    except (GuardRejection, SQLExecutionError) as exc:
        message = getattr(exc, "message", str(exc))
        return {"last_error": message}
    audit = {
        "action": result.guard.audit_action,
        "severity": result.guard.audit_severity,
        "rewrote_company_id": result.guard.rewrote_company_id,
        "rewritten_from_values": list(result.guard.rewritten_from_values),
        "injected_company_id_filters": result.guard.injected_company_id_filters,
        "enforced_limit": result.guard.enforced_limit,
        "tables": sorted(result.guard.tables),
    }
    return {
        "sql": result.guard.sql,
        "rows": result.rows,
        "rows_returned": result.rows_returned,
        "elapsed_ms": result.elapsed_ms,
        "audit": audit,
        "last_error": None,
    }


async def _interpret_node(state: SQLAnalystState) -> SQLAnalystState:
    rows = state.get("rows") or []
    rows_returned = int(state.get("rows_returned", 0))
    if rows_returned == 0:
        return {"summary": _EMPTY_RESULT_FALLBACK, "error": None}

    system_prompt, user_template = load_two_section_prompt(_INTERPRET_PROMPT_FILE)
    user_prompt = user_template.format(
        question=state["question"],
        sql=state.get("sql", ""),
        rows_returned=rows_returned,
        shown=min(_INTERPRET_ROW_CAP, rows_returned),
        rows_table=_render_rows_table(rows, _INTERPRET_ROW_CAP),
    )
    system_filled = system_prompt.format(user_role=state["user_role"])
    # The interpret step produces the user-facing answer in single-source mode;
    # tagging lets Chainlit stream its tokens. In route=both the parent
    # Finalize-combined LLM is also tagged and its stream replaces this one.
    llm = make_llm("specialist", tags=[FINAL_ANSWER_TAG])
    response = await invoke_llm(llm, [SystemMessage(system_filled), HumanMessage(user_prompt)])
    return {"summary": str(response.content).strip(), "error": None}


async def _fail_node(state: SQLAnalystState) -> SQLAnalystState:
    error = state.get("last_error") or "unknown error"
    summary = _FAIL_FALLBACK.format(
        attempts=state.get("attempts", state.get("attempt", 0)),
        error=error,
    )
    return {"summary": summary, "error": error}


def _route_after_execute(state: SQLAnalystState) -> str:
    settings = get_settings()
    if state.get("last_error") is None:
        return "interpret"
    if int(state.get("attempt", 0)) >= settings.sql_max_attempts:
        return "fail"
    return "generate_sql"


def build_sql_analyst_subgraph():
    graph = StateGraph(SQLAnalystState)
    graph.add_node("generate_sql", _generate_sql_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("interpret", _interpret_node)
    graph.add_node("fail", _fail_node)
    graph.add_edge(START, "generate_sql")
    graph.add_edge("generate_sql", "execute")
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {"interpret": "interpret", "generate_sql": "generate_sql", "fail": "fail"},
    )
    graph.add_edge("interpret", END)
    graph.add_edge("fail", END)
    return graph.compile()
