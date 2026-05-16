"""Debug CLI for the parent agent graph (Supervisor + routes + Finalize).

Run from repo root with ``PYTHONPATH=.`` so the ``app.*`` package is importable
(same convention as ``scripts/sql_analyst.py`` and ``scripts/docs_search.py``):

    PYTHONPATH=. uv run python scripts/chat.py \\
        --question "Сколько потратили на командировки за последние 90 дней?" \\
        --role finance_manager \\
        --company-id 1

Multi-turn (I-07): reuse the same ``--thread-id`` between calls and the
``PostgresSaver`` resumes the conversation log. The follow-up message goes
through the ``condense_question`` node, which rewrites references like
"а за Q2?" into a standalone question before routing.
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from decimal import Decimal

from app.db.checkpointer import checkpointer_lifespan
from app.graph.build import build_agent_graph
from app.services.chat_service import ChatService

_VALID_ROLES = ("cfo", "finance_manager", "accountant", "auditor")


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


async def _run(question: str, role: str, company_id: int, thread_id: str | None) -> dict:
    async with checkpointer_lifespan() as (_pool, saver):
        graph = build_agent_graph(checkpointer=saver)
        service = ChatService(graph)
        result = await service.ask(
            question=question,
            user_role=role,  # type: ignore[arg-type]
            company_id=company_id,
            thread_id=thread_id,
        )
        return asdict(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the parent agent graph end-to-end via ChatService."
    )
    parser.add_argument("--question", required=True, help="user question (Russian)")
    parser.add_argument("--role", required=True, choices=_VALID_ROLES)
    parser.add_argument(
        "--company-id",
        required=True,
        type=int,
        dest="company_id",
        help="tenant company_id from companies.id",
    )
    parser.add_argument("--thread-id", default=None, dest="thread_id")
    args = parser.parse_args()

    result = asyncio.run(_run(args.question, args.role, args.company_id, args.thread_id))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
