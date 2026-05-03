"""SQL Analyst subgraph: generate → execute (guarded) → interpret, with ReAct retry.

Four nodes:

* ``generate_sql`` — LLM produces one Postgres SELECT given the role-scoped
  schema and (on retries) the previous SQL + error.
* ``execute`` — ``execute_guarded`` runs the SQL through ``sql_guard`` and
  ``audit_log``. Guard rejections and DB errors both feed the retry loop.
* ``interpret`` — second LLM call summarises the resulting rows in Russian.
* ``fail`` — terminal Russian fallback used when ``sql_max_attempts`` is
  exhausted.

The retry edge from ``execute`` goes back to ``generate_sql`` while
``attempt < max_attempts``; otherwise it routes to ``fail``. Output state
contract used by the parent graph in I-06:
    in:  ``question``, ``user_role``, ``company_id``, ``thread_id``
    out: ``summary``, ``sql``, ``rows``, ``rows_returned``, ``audit``,
         ``attempts``, ``error``
"""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.tools.schema_introspect import schema_for_role
from app.tools.sql_executor import (
    ExecutionResult,
    SQLExecutionError,
    execute_guarded,
)
from app.tools.sql_guard import GuardRejection

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_GENERATE_PROMPT = _PROMPTS_DIR / "sql_analyst_generate.txt"
_INTERPRET_PROMPT = _PROMPTS_DIR / "sql_analyst_interpret.txt"

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER ==="

_INTERPRET_ROW_CAP = 50

_FAIL_FALLBACK = (
    "Не удалось получить ответ из базы за {attempts} попыт(ок). Последняя ошибка: {error}"
)
_EMPTY_RESULT_FALLBACK = "По заданным условиям записей не найдено."


class SQLAnalystState(TypedDict, total=False):
    question: str
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


def _load_prompt(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    if _SYSTEM_MARKER not in raw or _USER_MARKER not in raw:
        raise ValueError(f"{path} must contain both {_SYSTEM_MARKER!r} and {_USER_MARKER!r}")
    _, _, after_system = raw.partition(_SYSTEM_MARKER)
    system_part, _, user_part = after_system.partition(_USER_MARKER)
    return system_part.strip(), user_part.strip()


def _make_llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_specialist_model,
        temperature=0,
        timeout=settings.llm_request_timeout,
    )


def _strip_sql_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip().rstrip(";").strip()


def _format_retry_block(last_sql: str | None, last_error: str | None) -> str:
    if not last_sql or not last_error:
        return ""
    return (
        "Previous attempt failed.\n"
        "Previous SQL:\n"
        f"{last_sql}\n\n"
        "Error:\n"
        f"{last_error}\n\n"
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
    system_prompt, user_template = _load_prompt(_GENERATE_PROMPT)
    schema = schema_for_role(state["user_role"])
    user_prompt = user_template.format(
        schema=schema,
        question=state["question"],
        retry_block=_format_retry_block(state.get("candidate_sql"), state.get("last_error")),
    )
    system_filled = system_prompt.format(
        user_role=state["user_role"],
        company_id=state["company_id"],
    )
    llm = _make_llm()
    response = await llm.ainvoke([SystemMessage(system_filled), HumanMessage(user_prompt)])
    candidate = _strip_sql_fences(str(response.content))
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

    system_prompt, user_template = _load_prompt(_INTERPRET_PROMPT)
    user_prompt = user_template.format(
        question=state["question"],
        sql=state.get("sql", ""),
        rows_returned=rows_returned,
        shown=min(_INTERPRET_ROW_CAP, rows_returned),
        rows_table=_render_rows_table(rows, _INTERPRET_ROW_CAP),
    )
    system_filled = system_prompt.format(user_role=state["user_role"])
    llm = _make_llm()
    response = await llm.ainvoke([SystemMessage(system_filled), HumanMessage(user_prompt)])
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
