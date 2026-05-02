"""Debug CLI for the DocsResearcher subgraph.

Run from repo root, e.g.:

    uv run python scripts/docs_search.py \\
        --question "Какие лимиты на корпкарту топ-менеджера?" \\
        --role finance_manager \\
        --company-id 1
"""

import argparse
import asyncio
import json

from app.graph.docs_researcher import build_docs_researcher_subgraph


_VALID_ROLES = ("cfo", "finance_manager", "accountant", "auditor")


async def _run(question: str, role: str, company_id: int) -> dict:
    graph = build_docs_researcher_subgraph()
    return await graph.ainvoke(
        {"question": question, "user_role": role, "company_id": company_id}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DocsResearcher subgraph end-to-end."
    )
    parser.add_argument("--question", required=True, help="user question (Russian)")
    parser.add_argument("--role", required=True, choices=_VALID_ROLES)
    parser.add_argument(
        "--company-id", required=True, type=int, dest="company_id",
        help="tenant company_id from companies.id",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args.question, args.role, args.company_id))

    print("=" * 80)
    print(f"Tenant slug: {result.get('tenant_slug')}")
    print(f"Candidates after RRF: {len(result.get('candidates', []))}")
    print(f"Top chunks (post-rerank): {len(result.get('top_chunks', []))}")
    print()
    print("Summary:")
    print("-" * 80)
    print(result.get("summary", ""))
    print()
    print("Sources:")
    print("-" * 80)
    print(json.dumps(result.get("sources", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
