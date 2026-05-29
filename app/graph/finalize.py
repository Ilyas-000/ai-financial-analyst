"""Finalize node — aggregates specialist outputs into the final ``AgentState``.

* **Single-source**: the specialist already produced a polished summary in its
  own ``interpret`` / ``summarize`` step, so Finalize does NOT call an LLM —
  it picks the summary, normalises sources, and optionally builds a draft action.
* **Combined** (``route == "both"``, I-08): both summaries present, so a
  writer-LLM fuses them into one Russian answer; sources from both are merged.
"""

from datetime import date
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.llm import FINAL_ANSWER_TAG, invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState, effective_question
from app.tools.draft_action_builder import (
    DraftAction,
    DraftActionError,
    build_export_report,
    build_open_ticket,
)

logger = structlog.get_logger(__name__)

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

    Only kinds fillable from question + role + date are built; ``prepare_act`` /
    ``highlight_discrepancy`` need entity extraction we don't have yet (TD-11),
    so they're skipped gracefully (the hint stays in state for tracing).
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
            logger.info("draft_action_skipped_needs_entities", action_kind=kind)
            return None
    except DraftActionError as exc:
        logger.warning("draft_action_build_failed", error=str(exc))
        return None
    return action.as_payload()


async def _synthesise_combined(*, question: str, sql_summary: str, docs_summary: str) -> str:
    system_template, user_template = load_two_section_prompt(_COMBINED_PROMPT_FILE)
    user_prompt = user_template.format(
        question=question,
        sql_summary=sql_summary,
        docs_summary=docs_summary,
    )
    # ``final_answer`` tag — Chainlit streams this LLM's tokens in route=both.
    # In single-source paths Finalize doesn't call an LLM at all (specialist
    # tokens are streamed instead — see sql_analyst / docs_researcher).
    llm = make_llm("writer", tags=[FINAL_ANSWER_TAG])
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
            logger.warning("combined_synthesis_empty_falling_back_to_concat")
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
