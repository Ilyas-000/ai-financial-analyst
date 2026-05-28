"""Eval CLI runner.

Usage::

    uv run python -m eval.run_eval                       # full run
    uv run python -m eval.run_eval --only route_accuracy # smoke
    uv run python -m eval.run_eval --only ragas          # only LLM-as-judge metrics
    uv run python -m eval.run_eval --tags cross_tenant   # subset by tag
    uv run python -m eval.run_eval --max-cases 5         # sample

Boots the same agent graph the FastAPI app uses (``build_agent_graph`` +
``PostgresSaver``), drives each golden case through ``ChatService.ask``,
then reads the per-thread ``audit_log`` rows to compute SQL-side metrics
without needing the SQL itself to surface through ``ChatResult``.

Writes ``eval/reports/run_{utc_timestamp}.json`` and prints a summary.
"""

import argparse
import asyncio
import time
import uuid
from pathlib import Path

import structlog
from langchain_core.runnables import Runnable
from sqlalchemy import text

from app.config import get_settings
from app.db.checkpointer import checkpointer_lifespan
from app.db.session import get_sessionmaker
from app.graph.build import build_agent_graph
from app.graph.llm import make_llm
from app.observability.logging import configure_logging
from app.services.chat_service import ChatResult, ChatService
from app.services.tenants_view import list_tenants_for_ui
from eval.cases import GoldenCase, load_cases
from eval.metrics import (
    AuditRow,
    CaseRun,
    CaseScores,
    aggregate,
    aggregate_by_tag,
    answer_contains,
    answer_match_score,
    context_precision_score,
    context_recall,
    draft_action_accuracy,
    faithfulness_score,
    forbidden_check,
    route_match,
    sources_in_range,
    sql_correctness,
    tool_usage_accuracy,
)
from eval.metrics import answer_relevance_score as relevance_score
from eval.report import RunReport, build_case_report, make_run_id, print_summary, write_report

logger = structlog.get_logger(__name__)

_GOLDEN_DEFAULT = Path(__file__).resolve().parent / "golden.jsonl"

_PROFILES = ("all", "route_accuracy", "ragas")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden eval suite.")
    parser.add_argument("--golden", type=Path, default=_GOLDEN_DEFAULT)
    parser.add_argument("--only", choices=_PROFILES, default="all")
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated tags; cases must match at least one. Empty = all.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="0 = run all cases after tag filter."
    )
    return parser.parse_args()


def _filter_cases(cases: list[GoldenCase], tags_csv: str, max_cases: int) -> list[GoldenCase]:
    if tags_csv:
        wanted = {tag.strip() for tag in tags_csv.split(",") if tag.strip()}
        cases = [case for case in cases if wanted.intersection(case.tags)]
    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


async def _resolve_tenants() -> dict[str, int]:
    options = await list_tenants_for_ui()
    return {option.slug: option.company_id for option in options}


async def _fetch_audit_rows(thread_id: str) -> list[AuditRow]:
    """All audit_log rows whose ``details.thread_id`` matches the eval run."""
    sm = get_sessionmaker()
    query = text(
        """
        SELECT action, severity, sql_text
        FROM audit_log
        WHERE details ->> 'thread_id' = :thread_id
        ORDER BY created_at ASC
        """
    )
    async with sm() as session:
        result = await session.execute(query, {"thread_id": thread_id})
        return [
            AuditRow(action=row.action, severity=row.severity, sql_text=row.sql_text)
            for row in result.mappings()
        ]


async def _run_case(
    service: ChatService,
    case: GoldenCase,
    company_id: int,
) -> CaseRun:
    thread_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    error: str | None = None
    try:
        result = await service.ask(
            question=case.question,
            user_role=case.user_role,
            company_id=company_id,
            thread_id=thread_id,
        )
    except Exception as exc:
        logger.warning("eval_case_failed", case_id=case.id, error=str(exc))
        error = f"{type(exc).__name__}: {exc}"
        result = ChatResult(
            answer="",
            sources=[],
            suggested_action=None,
            route="error",
            route_reasoning=None,
            thread_id=thread_id,
            errors=[error],
        )
    elapsed = time.monotonic() - started
    audit_rows = await _fetch_audit_rows(thread_id)
    return CaseRun(
        case=case,
        company_id=company_id,
        result=result,
        audit_rows=audit_rows,
        latency_s=elapsed,
        error=error,
    )


def _custom_scores(run: CaseRun) -> CaseScores:
    return CaseScores(
        route_accuracy=route_match(run),
        sql_correctness=sql_correctness(run),
        tool_usage_accuracy=tool_usage_accuracy(run),
        draft_action_accuracy=draft_action_accuracy(run),
        forbidden_check=forbidden_check(run),
        answer_contains=answer_contains(run),
        sources_in_range=sources_in_range(run),
        context_recall=context_recall(run),
        latency_s=run.latency_s,
    )


async def _judge_scores(scores: CaseScores, run: CaseRun, judge_llm: Runnable) -> CaseScores:
    """Augment ``scores`` with LLM-as-judge metrics. Failures degrade to ``None``."""
    try:
        scores.faithfulness = await faithfulness_score(run, judge_llm)
    except Exception as exc:
        logger.warning("judge_failed", metric="faithfulness", case_id=run.case.id, error=str(exc))
    try:
        scores.answer_relevance = await relevance_score(run, judge_llm)
    except Exception as exc:
        logger.warning(
            "judge_failed", metric="answer_relevance", case_id=run.case.id, error=str(exc)
        )
    try:
        scores.context_precision = await context_precision_score(run, judge_llm)
    except Exception as exc:
        logger.warning(
            "judge_failed", metric="context_precision", case_id=run.case.id, error=str(exc)
        )
    try:
        scores.answer_match = await answer_match_score(run, judge_llm)
    except Exception as exc:
        logger.warning("judge_failed", metric="answer_match", case_id=run.case.id, error=str(exc))
    return scores


async def _verify_tenants_table() -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        result = await session.execute(text("SELECT COUNT(*) AS n FROM companies"))
        n = int(result.scalar() or 0)
    if n == 0:
        raise RuntimeError("companies table is empty — run `make migrate && make seed` first.")


async def run(args: argparse.Namespace) -> RunReport:
    configure_logging(get_settings())
    cases = _filter_cases(load_cases(args.golden), args.tags, args.max_cases)
    if not cases:
        raise SystemExit("no cases matched the given filter")
    await _verify_tenants_table()
    tenants = await _resolve_tenants()

    started = make_run_id()
    judge_llm = make_llm("writer", temperature=0.0)
    case_reports = []
    all_scores: list[CaseScores] = []
    skipped = 0

    async with checkpointer_lifespan() as (_pool, saver):
        graph = build_agent_graph(checkpointer=saver)
        service = ChatService(graph)
        for case in cases:
            company_id = tenants.get(case.tenant_slug)
            if company_id is None:
                logger.warning("skip_case_unknown_tenant", tenant_slug=case.tenant_slug)
                skipped += 1
                continue
            logger.info("eval_case_start", case_id=case.id, tenant_slug=case.tenant_slug)
            run_record = await _run_case(service, case, company_id)
            if args.only == "ragas":
                scores = CaseScores(latency_s=run_record.latency_s)
                scores = await _judge_scores(scores, run_record, judge_llm)
            elif args.only == "route_accuracy":
                scores = CaseScores(
                    route_accuracy=route_match(run_record), latency_s=run_record.latency_s
                )
            else:
                scores = _custom_scores(run_record)
                scores = await _judge_scores(scores, run_record, judge_llm)
            all_scores.append(scores)
            case_reports.append(
                build_case_report(
                    case=case,
                    observed_route=run_record.result.route,
                    answer=run_record.result.answer,
                    sources_count=len(run_record.result.sources),
                    suggested_action_kind=(run_record.result.suggested_action or {}).get("kind"),
                    error=run_record.error,
                    latency_s=run_record.latency_s,
                    scores=scores,
                )
            )

    finished = make_run_id()
    aggregates = aggregate(all_scores)
    by_tag = {
        metric: aggregate_by_tag(cases, all_scores, metric)
        for metric in (
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
    }
    by_tag = {metric: data for metric, data in by_tag.items() if data}

    return RunReport(
        run_id=started,
        started_at=started,
        finished_at=finished,
        profile=args.only,
        total_cases=len(cases),
        skipped=skipped,
        aggregates=aggregates,
        by_tag=by_tag,
        cases=case_reports,
    )


def main() -> None:
    args = _parse_args()
    report = asyncio.run(run(args))
    path = write_report(report)
    print_summary(report)
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
