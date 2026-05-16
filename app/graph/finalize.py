"""Finalize node — aggregates specialist outputs into the final ``AgentState``.

I-06 scope: only single-source paths (``sql_analyst`` or ``docs_researcher``).
The specialist already produced a polished Russian summary in its own
``interpret`` / ``summarize`` step, so Finalize does NOT call an LLM here —
it just picks the right summary, normalises ``sources``, and (if Supervisor
hinted ``suggest_action_kind``) builds a ``suggested_action`` payload via
``draft_action_builder``.

When parallel ``route=both`` is added in I-08, Finalize will gain a
writer-LLM step that synthesises a combined answer; the contract this node
returns stays the same.
"""

import logging
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage

from app.graph.state import AgentState, effective_question
from app.tools.draft_action_builder import (
    DraftAction,
    DraftActionError,
    build_export_report,
    build_open_ticket,
)

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = "Не удалось сформировать ответ по вашему вопросу. Уточните формулировку."


def _normalise_sql_sources(sql_result: dict[str, Any]) -> list[dict[str, Any]]:
    sql = sql_result.get("sql")
    if not sql:
        return []
    return [
        {
            "type": "sql",
            "sql": sql,
            "rows_returned": sql_result.get("rows_returned", 0),
            "attempts": sql_result.get("attempts"),
            "audit": sql_result.get("audit"),
        }
    ]


def _normalise_doc_sources(docs_result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sources = docs_result.get("sources") or []
    return [{"type": "doc", **source} for source in raw_sources]


def _build_action(state: AgentState) -> dict[str, Any] | None:
    """Build a draft ``suggested_action`` from Supervisor's hint, if possible.

    In I-06 we only build payloads we can fill from the question + role +
    today's date. ``prepare_act`` and ``highlight_discrepancy`` need entities
    we don't yet extract, so we skip them gracefully (Supervisor's hint is
    preserved in state for tracing).
    """
    kind = state.get("suggest_action_kind")
    if not kind:
        return None
    today = date.today()
    period_start = today.replace(day=1)
    period_end = today
    question = effective_question(state).strip()
    try:
        action: DraftAction
        if kind == "export_report":
            action = build_export_report(
                report_type="ad_hoc",
                period_start=period_start,
                period_end=period_end,
                currency="RUB",
                fmt="csv",
            )
        elif kind == "open_ticket":
            topic = question[:80] if question else "Эскалация по запросу пользователя"
            action = build_open_ticket(
                topic=topic,
                severity="medium",
                summary=question or "Подробности уточнить у пользователя.",
            )
        else:
            logger.info(
                "skip suggested_action build for kind=%s (needs entity extraction)",
                kind,
            )
            return None
    except DraftActionError as exc:
        logger.warning("draft action build failed: %s", exc)
        return None
    return action.as_payload()


def finalize_node(state: AgentState) -> AgentState:
    sql_result = state.get("sql_result") or {}
    docs_result = state.get("docs_result") or {}

    parts: list[str] = []
    if sql_result.get("summary"):
        parts.append(str(sql_result["summary"]).strip())
    if docs_result.get("summary"):
        parts.append(str(docs_result["summary"]).strip())

    final_answer = "\n\n".join(p for p in parts if p) or _FALLBACK_ANSWER

    sources: list[dict[str, Any]] = []
    if sql_result:
        sources.extend(_normalise_sql_sources(sql_result))
    if docs_result:
        sources.extend(_normalise_doc_sources(docs_result))

    suggested_action = _build_action(state)

    return {
        "final_answer": final_answer,
        "sources": sources,
        "suggested_action": suggested_action,
        "messages": [AIMessage(content=final_answer)],
    }
