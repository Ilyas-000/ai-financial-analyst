"""Per-case eval metrics + small aggregators.

Two metric families:

* **Custom (sync, observable)** — derived from ``ChatResult`` + the
  ``audit_log`` rows the case produced. No LLM calls. Cheap; always run.
* **LLM-as-judge (async, gated)** — pass-through to a writer LLM that
  scores ``faithfulness`` / ``answer_relevance`` / ``context_precision``
  and a generative ``answer_match``. Only run when the runner selects the
  ``ragas`` or ``judge`` profile.

A per-case metric returns ``float`` in ``[0.0, 1.0]`` or ``None`` if the
case is not applicable (e.g. ``sql_correctness`` on a RAG-only case).
``None`` is excluded from aggregation.
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.llm import LLMRunnable, invoke_llm
from app.services.chat_service import ChatResult
from eval.cases import GoldenCase


@dataclass(frozen=True)
class AuditRow:
    action: str
    severity: str
    sql_text: str | None


@dataclass(frozen=True)
class CaseRun:
    case: GoldenCase
    company_id: int
    result: ChatResult
    audit_rows: list[AuditRow]
    latency_s: float
    error: str | None = None


@dataclass
class CaseScores:
    route_accuracy: float | None = None
    sql_correctness: float | None = None
    tool_usage_accuracy: float | None = None
    draft_action_accuracy: float | None = None
    forbidden_check: float | None = None
    answer_contains: float | None = None
    sources_in_range: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    answer_match: float | None = None
    latency_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


# --- Custom metrics ---------------------------------------------------------


def route_match(run: CaseRun) -> float:
    return 1.0 if run.result.route in set(run.case.expected_route) else 0.0


def sql_correctness(run: CaseRun) -> float | None:
    if "sql_executor" not in run.case.expected_tools:
        return None
    success_actions = {"sql_executed", "tenancy_rewrite"}
    success_rows = [r for r in run.audit_rows if r.action in success_actions]
    if not success_rows:
        return 0.0
    sql = (success_rows[-1].sql_text or "").lower()
    for needle in run.case.expected_sql_contains:
        if needle.lower() not in sql:
            return 0.0
    return 1.0


def tool_usage_accuracy(run: CaseRun) -> float:
    """Observed vs expected tool set. Currently tracks ``sql_executor`` and
    ``retriever`` — FX / calculator are deferred until they are wired into
    the graph (see tech_debt)."""
    observed: set[str] = set()
    sql_ok = any(r.action in {"sql_executed", "tenancy_rewrite"} for r in run.audit_rows)
    if sql_ok:
        observed.add("sql_executor")
    has_doc_source = any(s.get("type") == "doc" for s in run.result.sources)
    if has_doc_source or run.result.route in {"docs_researcher", "both"}:
        observed.add("retriever")
    return 1.0 if observed == set(run.case.expected_tools) else 0.0


def draft_action_accuracy(run: CaseRun) -> float:
    expected = run.case.expected_suggested_action_kind
    action = run.result.suggested_action
    actual = (action or {}).get("kind") if action else None
    if expected == "none":
        return 1.0 if not actual else 0.0
    return 1.0 if actual == expected else 0.0


def forbidden_check(run: CaseRun) -> float | None:
    if not run.case.forbidden_in_answer:
        return None
    answer_lower = run.result.answer.lower()
    for forbidden in run.case.forbidden_in_answer:
        if forbidden.lower() in answer_lower:
            return 0.0
    return 1.0


def answer_contains(run: CaseRun) -> float | None:
    if not run.case.expected_answer_contains:
        return None
    answer_lower = run.result.answer.lower()
    for needle in run.case.expected_answer_contains:
        if needle.lower() not in answer_lower:
            return 0.0
    return 1.0


def sources_in_range(run: CaseRun) -> float:
    n = len(run.result.sources)
    if n < run.case.expected_sources_min or n > run.case.expected_sources_max:
        return 0.0
    return 1.0


def context_recall(run: CaseRun) -> float | None:
    expected = set(run.case.expected_doc_ids)
    if not expected:
        return None
    retrieved: set[str] = set()
    for source in run.result.sources:
        if source.get("type") != "doc":
            continue
        doc_id = source.get("doc_id") or source.get("id")
        if doc_id:
            retrieved.add(str(doc_id))
    if not expected:
        return 1.0
    return len(expected & retrieved) / len(expected)


# --- LLM-as-judge ------------------------------------------------------------


class _JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=400)


_JUDGE_SYSTEM = (
    "You are a strict evaluator for a Russian-language financial-analyst "
    "agent. Score each criterion on a 0.0–1.0 scale. Return JSON with "
    "fields {{score: float, reason: string}}. Be concise."
)


async def _judge(llm: LLMRunnable, user_prompt: str) -> _JudgeVerdict:
    structured = llm.with_structured_output(_JudgeVerdict)
    return await invoke_llm(structured, [SystemMessage(_JUDGE_SYSTEM), HumanMessage(user_prompt)])


def _context_blob(run: CaseRun) -> str:
    parts: list[str] = []
    for source in run.result.sources:
        if source.get("type") == "doc":
            label = source.get("doc_title") or source.get("doc_id") or "doc"
            section = source.get("section") or ""
            parts.append(f"[doc:{label}{(' / ' + section) if section else ''}]")
        elif source.get("type") == "sql":
            rows = source.get("rows_returned", 0)
            parts.append(f"[sql:{rows} rows]")
    return "; ".join(parts) or "(no sources)"


async def faithfulness_score(run: CaseRun, llm: LLMRunnable) -> float | None:
    if not run.result.sources:
        return None
    prompt = (
        "Question: " + run.case.question + "\n\n"
        "Sources: " + _context_blob(run) + "\n\n"
        "Answer:\n" + run.result.answer + "\n\n"
        "Score the answer's faithfulness to the listed sources. 1.0 = every "
        "claim is grounded in sources, 0.0 = answer fabricates facts not in "
        "sources. Ignore stylistic differences."
    )
    verdict = await _judge(llm, prompt)
    return float(verdict.score)


async def answer_relevance_score(run: CaseRun, llm: LLMRunnable) -> float | None:
    if not run.case.expected_answer_contains and not run.result.answer:
        return None
    prompt = (
        "Question: " + run.case.question + "\n\n"
        "Answer:\n" + run.result.answer + "\n\n"
        "Score how directly the answer addresses the question. 1.0 = "
        "directly answers, 0.0 = unrelated / evasive."
    )
    verdict = await _judge(llm, prompt)
    return float(verdict.score)


async def context_precision_score(run: CaseRun, llm: LLMRunnable) -> float | None:
    docs = [s for s in run.result.sources if s.get("type") == "doc"]
    if not docs:
        return None
    relevant = 0
    for source in docs:
        label = source.get("doc_title") or source.get("doc_id") or "doc"
        section = source.get("section") or ""
        prompt = (
            "Question: " + run.case.question + "\n\n"
            f"Retrieved chunk: {label}" + (f" / {section}" if section else "") + "\n\n"
            "Score whether this chunk is relevant to answering the question. "
            "1.0 = relevant, 0.0 = irrelevant."
        )
        verdict = await _judge(llm, prompt)
        if verdict.score >= 0.5:
            relevant += 1
    return relevant / len(docs)


async def answer_match_score(run: CaseRun, llm: LLMRunnable) -> float | None:
    if not run.case.expected_answer_contains:
        return None
    expected = ", ".join(run.case.expected_answer_contains)
    prompt = (
        "Question: " + run.case.question + "\n\n"
        f"Expected the answer to cover these themes/keywords: {expected}\n\n"
        "Actual answer:\n" + run.result.answer + "\n\n"
        "Score 1.0 if every expected theme is present (synonyms allowed). "
        "Score 0.0 if any required theme is missing."
    )
    verdict = await _judge(llm, prompt)
    return float(verdict.score)


# --- Aggregation -------------------------------------------------------------


def aggregate(scores: list[CaseScores]) -> dict[str, float]:
    out: dict[str, float] = {}
    metric_names = (
        "route_accuracy",
        "sql_correctness",
        "tool_usage_accuracy",
        "draft_action_accuracy",
        "forbidden_check",
        "answer_contains",
        "sources_in_range",
        "context_recall",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "answer_match",
    )
    for name in metric_names:
        values = [getattr(s, name) for s in scores if getattr(s, name) is not None]
        if values:
            out[name] = mean(values)
    latencies = [s.latency_s for s in scores if s.latency_s > 0]
    if latencies:
        sorted_lat = sorted(latencies)

        def percentile(p: float) -> float:
            if not sorted_lat:
                return 0.0
            k = max(0, min(len(sorted_lat) - 1, int(round(p * (len(sorted_lat) - 1)))))
            return float(sorted_lat[k])

        out["latency_p50"] = percentile(0.50)
        out["latency_p95"] = percentile(0.95)
    return out


def aggregate_by_tag(
    cases: list[GoldenCase],
    scores: list[CaseScores],
    metric: str,
) -> dict[str, float]:
    by_tag: dict[str, list[float]] = {}
    for case, score in zip(cases, scores, strict=True):
        value = getattr(score, metric, None)
        if value is None:
            continue
        for tag in case.tags:
            by_tag.setdefault(tag, []).append(value)
    return {tag: mean(values) for tag, values in by_tag.items() if values}
