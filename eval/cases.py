"""Golden eval case schema and JSONL I/O.

A case is one line in ``eval/golden.jsonl``. The generator writes them, the
runner reads them. Fields are anchored to signals observable in one run:

* ``expected_route`` — list of accepted ``ChatResult.route`` values (a list,
  not a single string, so cross-tenant cases can pass when Supervisor stayed
  on ``sql_analyst`` *or* when ``output_guard`` rewrote it to ``blocked``).
* ``expected_sql_contains`` — substrings the SQL stored in ``audit_log`` for
  the case's ``thread_id`` must contain (case-insensitive).
* ``expected_doc_ids`` — RAG doc ids that should appear in
  ``ChatResult.sources`` with ``type == "doc"`` (used by context_recall).
* ``forbidden_in_answer`` — substrings that must NOT appear in the final
  answer. The cross-tenant invariant lives here.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

UserRole = Literal["cfo", "finance_manager", "accountant", "auditor"]
TenantSlug = Literal["acme", "ostrovok", "checkscan"]
ExpectedRoute = Literal[
    "sql_analyst",
    "docs_researcher",
    "both",
    "direct_answer",
    "clarify",
    "blocked",
]
ExpectedTool = Literal["sql_executor", "retriever"]
ActionKind = Literal[
    "export_report",
    "open_ticket",
    "prepare_act",
    "highlight_discrepancy",
    "none",
]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    user_role: UserRole
    tenant_slug: TenantSlug
    expected_route: list[ExpectedRoute]
    expected_tools: list[ExpectedTool]
    expected_sql_contains: list[str]
    expected_answer_contains: list[str]
    expected_sources_min: int
    expected_sources_max: int
    expected_suggested_action_kind: ActionKind
    expected_doc_ids: list[str] = field(default_factory=list)
    forbidden_in_answer: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


def load_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                GoldenCase(
                    id=row["id"],
                    question=row["question"],
                    user_role=row["user_role"],
                    tenant_slug=row["tenant_slug"],
                    expected_route=list(row["expected_route"]),
                    expected_tools=list(row.get("expected_tools", [])),
                    expected_sql_contains=list(row.get("expected_sql_contains", [])),
                    expected_answer_contains=list(row.get("expected_answer_contains", [])),
                    expected_sources_min=int(row.get("expected_sources_min", 0)),
                    expected_sources_max=int(row.get("expected_sources_max", 999)),
                    expected_suggested_action_kind=row.get(
                        "expected_suggested_action_kind", "none"
                    ),
                    expected_doc_ids=list(row.get("expected_doc_ids", [])),
                    forbidden_in_answer=list(row.get("forbidden_in_answer", [])),
                    tags=list(row.get("tags", [])),
                    notes=row.get("notes"),
                )
            )
    return cases


def dump_cases(cases: list[GoldenCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
