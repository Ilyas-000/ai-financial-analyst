"""Finalize node — aggregates specialist outputs into the final ``AgentState``.

Two modes:

* **Single-source** (``route == "sql_analyst"`` or ``"docs_researcher"``):
  the specialist already produced a polished Russian summary in its own
  ``interpret`` / ``summarize`` step, so Finalize does NOT call an LLM here —
  it just picks the right summary, normalises ``sources``, and (if Supervisor
  hinted ``suggest_action_kind``) builds a ``suggested_action`` payload via
  ``draft_action_builder``.

* **Combined** (``route == "both"``, I-08): both ``sql_result`` and
  ``docs_result`` carry a non-empty summary. Finalize calls a writer-LLM with
  the question + both summaries to produce one coherent Russian answer (fact
  vs regulation tie-in). Sources from both branches are merged.
"""

import logging
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState, effective_question
from app.tools.draft_action_builder import (
    DraftAction,
    DraftActionError,
    build_export_report,
    build_open_ticket,
)

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = "Не удалось сформировать ответ по вашему вопросу. Уточните формулировку."
_COMBINED_PROMPT_FILE = "finalize_combined.txt"


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


async def _synthesise_combined(*, question: str, sql_summary: str, docs_summary: str) -> str:
    system_template, user_template = load_two_section_prompt(_COMBINED_PROMPT_FILE)
    user_prompt = user_template.format(
        question=question,
        sql_summary=sql_summary,
        docs_summary=docs_summary,
    )
    llm = make_llm("writer")
    response = await invoke_llm(llm, [SystemMessage(system_template), HumanMessage(user_prompt)])
    return str(response.content).strip()


async def finalize_node(state: AgentState) -> AgentState:
    sql_result = state.get("sql_result") or {}
    docs_result = state.get("docs_result") or {}

    sql_summary = str(sql_result.get("summary") or "").strip()
    docs_summary = str(docs_result.get("summary") or "").strip()

    if sql_summary and docs_summary:
        final_answer = await _synthesise_combined(
            question=effective_question(state),
            sql_summary=sql_summary,
            docs_summary=docs_summary,
        )
        if not final_answer:
            logger.warning("combined synthesis produced empty answer, falling back to concat")
            final_answer = f"{sql_summary}\n\n{docs_summary}"
    elif sql_summary or docs_summary:
        final_answer = sql_summary or docs_summary
    else:
        final_answer = _FALLBACK_ANSWER

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
