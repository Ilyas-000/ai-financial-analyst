"""Finalize node — aggregates specialist outputs into the final ``AgentState``.

* **Single-source**: the specialist already produced a polished summary in its
  own ``interpret`` / ``summarize`` step, so Finalize does NOT call an LLM —
  it picks the summary, normalises sources, and optionally builds a draft action.
* **Combined** (``route == "both"``, I-08): both summaries present, so a
  writer-LLM fuses them into one Russian answer; sources from both are merged.
"""

import re
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
    build_highlight_discrepancy,
    build_open_ticket,
    build_prepare_act,
)

logger = structlog.get_logger(__name__)

_FALLBACK_ANSWER = "Не удалось сформировать ответ по вашему вопросу. Уточните формулировку."
_COMBINED_PROMPT_FILE = "finalize_combined.txt"

# Russian legal-form prefix + up to three capitalised name tokens, e.g.
# "ООО Ромашка", "ИП Иванов", "ПАО «Сбербанк»". Quotes are stripped afterwards.
_COUNTERPARTY_RE = re.compile(
    r"\b(?:ООО|ОАО|ПАО|АО|ЗАО|ИП|НКО)\s+[«\"„]?[A-ZА-ЯЁ][\w\-]*(?:\s+[A-ZА-ЯЁ][\w\-]*){0,2}"
)
# A monetary figure anchored to a ruble marker (``\u20bd`` / ``\u0440\u0443\u0431``), e.g.
# "612 340,55 \u20bd" / "500 000 \u0440\u0443\u0431.". Thousands use space/nbsp (not arbitrary
# whitespace, so it can't gobble across a newline). Anchoring to the currency
# avoids grabbing a bare year or count earlier in the sentence
# (".. 2025 .. 612 340,55 RUB" -> "612 340,55", not "2025").
_AMOUNT_RE = re.compile(
    r"(\d[\d \u00a0]*(?:[.,]\d+)?)\s*(?:\u20bd|\u0440\u0443\u0431)", re.IGNORECASE
)
_QUOTE_CHARS = "«»\"„“”'"


def _extract_counterparty(question: str) -> str | None:
    match = _COUNTERPARTY_RE.search(question)
    if not match:
        return None
    name = "".join(ch for ch in match.group(0) if ch not in _QUOTE_CHARS)
    name = " ".join(name.split())
    return name or None


def _extract_payout_ids(sql_result: dict[str, Any]) -> list[int]:
    """Pull payout row identifiers from the SQL result rows (``payout_id``/``id``)."""
    ids: list[int] = []
    for row in sql_result.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for key in ("payout_id", "id"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            ids.append(value)
            break
    return ids


def _first_amount(text: str) -> str | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    return " ".join(match.group(1).split()) or None


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

    Entities are extracted deterministically (TD-11): ``prepare_act`` needs a
    counterparty (parsed from the question) and payout ids (from the SQL rows);
    ``highlight_discrepancy`` needs the expected (docs) and actual (SQL) figures
    pulled from the two specialist summaries. When the required entities are
    absent the kind is skipped gracefully — the hint stays in state for tracing.
    """
    kind = state.get("suggest_action_kind")
    if not kind:
        return None
    today = date.today()
    period_start = today.replace(day=1)
    period_end = today
    question = effective_question(state).strip()
    sql_result = state.get("sql_result") or {}
    docs_result = state.get("docs_result") or {}
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
        elif kind == "prepare_act":
            counterparty = _extract_counterparty(question)
            payout_ids = _extract_payout_ids(sql_result)
            if not counterparty or not payout_ids:
                logger.info(
                    "draft_action_skipped_missing_entities",
                    action_kind=kind,
                    has_counterparty=bool(counterparty),
                    n_payout_ids=len(payout_ids),
                )
                return None
            action = build_prepare_act(
                counterparty_name=counterparty,
                period_start=period_start,
                period_end=period_end,
                payout_ids=payout_ids,
            )
        elif kind == "highlight_discrepancy":
            dimension = question[:80]
            actual = _first_amount(str(sql_result.get("summary") or ""))
            expected = _first_amount(str(docs_result.get("summary") or ""))
            if not dimension or expected is None or actual is None:
                logger.info(
                    "draft_action_skipped_missing_entities",
                    action_kind=kind,
                    has_expected=expected is not None,
                    has_actual=actual is not None,
                )
                return None
            action = build_highlight_discrepancy(
                dimension=dimension,
                expected=expected,
                actual=actual,
            )
        else:
            logger.info("draft_action_skipped_unknown_kind", action_kind=kind)
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
