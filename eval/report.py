"""Eval report shape + JSON writer.

The report is the persistent output of ``run_eval.py``. One JSON file per
run, written to ``eval/reports/run_{utc_timestamp}.json`` (gitignored).
The runner prints a short Markdown-ish summary alongside the file for
fast-scan in CI / terminal.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.cases import GoldenCase
from eval.metrics import CaseScores

_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass
class CaseReport:
    case_id: str
    tags: list[str]
    user_role: str
    tenant_slug: str
    question: str
    observed_route: str
    expected_route: list[str]
    answer_excerpt: str
    sources_count: int
    suggested_action_kind: str | None
    expected_action_kind: str
    error: str | None
    latency_s: float
    scores: dict[str, float | None]


@dataclass
class RunReport:
    run_id: str
    started_at: str
    finished_at: str
    profile: str
    total_cases: int
    skipped: int
    aggregates: dict[str, float]
    by_tag: dict[str, dict[str, float]]
    cases: list[CaseReport] = field(default_factory=list)


def build_case_report(
    case: GoldenCase,
    *,
    observed_route: str,
    answer: str,
    sources_count: int,
    suggested_action_kind: str | None,
    error: str | None,
    latency_s: float,
    scores: CaseScores,
) -> CaseReport:
    return CaseReport(
        case_id=case.id,
        tags=list(case.tags),
        user_role=case.user_role,
        tenant_slug=case.tenant_slug,
        question=case.question,
        observed_route=observed_route,
        expected_route=list(case.expected_route),
        answer_excerpt=answer[:300],
        sources_count=sources_count,
        suggested_action_kind=suggested_action_kind,
        expected_action_kind=case.expected_suggested_action_kind,
        error=error,
        latency_s=latency_s,
        scores={k: v for k, v in asdict(scores).items() if k != "notes"},
    )


def write_report(report: RunReport) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"run_{report.run_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, ensure_ascii=False, indent=2, default=_default)
    return path


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def print_summary(report: RunReport) -> None:
    print(f"\n=== eval run {report.run_id} — profile={report.profile} ===")
    print(f"cases: {report.total_cases}  skipped: {report.skipped}")
    print("\nAggregates:")
    for name, value in sorted(report.aggregates.items()):
        if name.startswith("latency_"):
            print(f"  {name:<26} {value:.2f}s")
        else:
            print(f"  {name:<26} {value:.3f}")
    print("\nRoute accuracy by tag:")
    for tag, value in sorted(report.by_tag.get("route_accuracy", {}).items()):
        print(f"  {tag:<26} {value:.3f}")
    cross = report.by_tag.get("forbidden_check", {}).get("cross_tenant")
    if cross is not None:
        print(f"\nCross-tenant forbidden_check (must be 1.0): {cross:.3f}")
