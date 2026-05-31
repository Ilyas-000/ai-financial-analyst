"""Reconstruct a human-readable agent trace from a finished ``ChatResult``.

The Chainlit UI shows these as collapsible ``cl.Step`` blocks so the user can
see *what the agent did* (which route the Supervisor chose, the SQL it ran, the
documents it retrieved) without opening Langfuse.

Why post-hoc (not live): the streaming model (I-11 R4) runs the graph to
completion and applies the output guard before the UI sees anything, so we
cannot surface live node tokens without fighting the cross-tenant guard.
Instead we rebuild the trace deterministically from the already-guarded
``ChatResult`` — pure, side-effect-free, and free of any ``chainlit`` import so
it stays testable and respects the UI import-isolation rule (``rules.md`` §2.1).
"""

from dataclasses import dataclass
from typing import Any

from app.services.chat_service import ChatResult

# Chainlit ``cl.Step`` type strings (plain strings — no chainlit import here).
# The renderer in ``chainlit_app.py`` passes ``step_type`` straight to
# ``cl.Step(type=...)``; values mirror Chainlit's built-in step kinds.
_TYPE_LLM = "llm"
_TYPE_TOOL = "tool"
_TYPE_RETRIEVAL = "retrieval"

_ROUTE_LABELS: dict[str, str] = {
    "sql_analyst": "Маршрут: SQL-аналитик (структурированные данные)",
    "docs_researcher": "Маршрут: поиск по документам (регламенты)",
    "both": "Маршрут: данные + документы",
    "direct_answer": "Маршрут: прямой ответ (без обращения к данным/документам)",
    "clarify": "Маршрут: уточняющий вопрос",
    "blocked": "Запрос заблокирован guardrail",
    "error": "Ошибка обработки запроса",
    "unknown": "Маршрут не определён",
}


@dataclass(frozen=True)
class TraceStep:
    """One renderable step: ``name`` + Markdown ``content`` + Chainlit type."""

    name: str
    content: str
    step_type: str = _TYPE_TOOL


def _routing_step(result: ChatResult) -> TraceStep:
    label = _ROUTE_LABELS.get(result.route, f"Маршрут: {result.route}")
    lines = [label]
    if result.route_reasoning:
        lines.append("")
        lines.append(result.route_reasoning.strip())
    return TraceStep(name="Supervisor", content="\n".join(lines), step_type=_TYPE_LLM)


def _sql_step(source: dict[str, Any]) -> TraceStep:
    sql = (source.get("sql") or "").strip()
    rows = source.get("rows_returned", 0)
    attempts = source.get("attempts")
    meta = [f"строк возвращено: {rows}"]
    if attempts:
        meta.append(f"попыток генерации SQL: {attempts}")
    content = ", ".join(meta)
    if sql:
        content += f"\n```sql\n{sql}\n```"
    return TraceStep(name="SQL Analyst", content=content, step_type=_TYPE_TOOL)


def _docs_step(sources: list[dict[str, Any]]) -> TraceStep:
    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        title = source.get("doc_title") or source.get("doc_id") or "документ"
        bits = [f"[{index}] {title}"]
        section = source.get("section")
        if section:
            bits.append(f"§ {section}")
        score = source.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            bits.append(f"score={score:.3f}")
        lines.append(" · ".join(bits))
    body = "\n".join(lines) or "Документы не найдены."
    content = f"найдено фрагментов: {len(sources)}\n\n{body}"
    return TraceStep(name="Docs Researcher", content=content, step_type=_TYPE_RETRIEVAL)


def build_trace_steps(result: ChatResult) -> list[TraceStep]:
    """Rebuild the agent's trace as an ordered list of renderable steps.

    Order mirrors execution: Supervisor first, then whichever specialists left
    evidence in ``sources`` (SQL before docs), then a guardrail/error step if
    the run carried any ``errors``. Specialist steps are driven by the actual
    ``sources`` rather than ``route`` so a ``both`` run with only one summary
    still shows exactly what ran.
    """
    steps: list[TraceStep] = [_routing_step(result)]

    for source in result.sources:
        if source.get("type") == "sql":
            steps.append(_sql_step(source))

    doc_sources = [s for s in result.sources if s.get("type") == "doc"]
    if doc_sources:
        steps.append(_docs_step(doc_sources))

    if result.errors:
        body = "\n".join(f"- {error}" for error in result.errors)
        steps.append(TraceStep(name="Guardrails / ошибки", content=body, step_type=_TYPE_TOOL))

    return steps
