"""Debug CLI for the SQL Analyst subgraph.

Run from repo root, e.g.:

    uv run python scripts/sql_analyst.py \\
        --question "Сколько потратили на командировки за последние 90 дней?" \\
        --role finance_manager \\
        --company-id 1
"""

import argparse
import asyncio
import json
from decimal import Decimal

from app.graph.sql_analyst import build_sql_analyst_subgraph
from app.services.chat_service import synth_user_id

_VALID_ROLES = ("cfo", "finance_manager", "accountant", "auditor")


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


async def _run(question: str, role: str, company_id: int, thread_id: str | None) -> dict:
    graph = build_sql_analyst_subgraph()
    return await graph.ainvoke(
        {
            "question": question,
            "user_id": synth_user_id(company_id, role),
            "user_role": role,
            "company_id": company_id,
            "thread_id": thread_id,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SQL Analyst subgraph end-to-end.")
    parser.add_argument("--question", required=True, help="user question (Russian)")
    parser.add_argument("--role", required=True, choices=_VALID_ROLES)
    parser.add_argument(
        "--company-id",
        required=True,
        type=int,
        dest="company_id",
        help="tenant company_id from companies.id",
    )
    parser.add_argument("--thread-id", default="cli", dest="thread_id")
    args = parser.parse_args()

    result = asyncio.run(_run(args.question, args.role, args.company_id, args.thread_id))

    print("=" * 80)
    print(f"attempts: {result.get('attempts')}")
    print()
    print("SQL (post-guard):")
    print("-" * 80)
    print(result.get("sql", "(none)"))
    print()
    print(
        f"rows_returned: {result.get('rows_returned', 0)}, "
        f"elapsed_ms: {result.get('elapsed_ms', 0)}"
    )
    print("audit:")
    print(json.dumps(result.get("audit", {}), ensure_ascii=False, indent=2))
    print()
    print("Summary:")
    print("-" * 80)
    print(result.get("summary", ""))
    if result.get("error"):
        print()
        print("Error trail:", result["error"])
    print()
    print("First rows:")
    print("-" * 80)
    head = result.get("rows", [])[:5]
    print(json.dumps(head, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
