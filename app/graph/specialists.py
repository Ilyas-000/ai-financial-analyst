"""Wrapper nodes that adapt specialist subgraphs to the parent ``AgentState``.

Each subgraph has its own state schema (``SQLAnalystState`` / ``DocsResearcherState``).
We invoke them with the projected input fields and route their output into
namespaced ``sql_result`` / ``docs_result`` slots on the parent state. Compiled
subgraph instances are cached so repeated invocations don't re-build the graph.
"""

from functools import lru_cache
from typing import Any

from app.graph.docs_researcher import build_docs_researcher_subgraph
from app.graph.sql_analyst import build_sql_analyst_subgraph
from app.graph.state import AgentState, effective_question


@lru_cache(maxsize=1)
def _sql_analyst_compiled():
    return build_sql_analyst_subgraph()


@lru_cache(maxsize=1)
def _docs_researcher_compiled():
    return build_docs_researcher_subgraph()


def _project_sql_result(sub_out: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": sub_out.get("summary"),
        "sql": sub_out.get("sql"),
        "rows": sub_out.get("rows"),
        "rows_returned": sub_out.get("rows_returned", 0),
        "audit": sub_out.get("audit"),
        "attempts": sub_out.get("attempts"),
        "error": sub_out.get("error"),
    }


def _project_docs_result(sub_out: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": sub_out.get("summary"),
        "sources": sub_out.get("sources") or [],
    }


async def sql_analyst_node(state: AgentState) -> AgentState:
    sub_in = {
        "question": effective_question(state),
        "user_id": state["user_id"],
        "user_role": state["user_role"],
        "company_id": state["company_id"],
        "thread_id": state.get("thread_id"),
    }
    sub_out = await _sql_analyst_compiled().ainvoke(sub_in)
    return {"sql_result": _project_sql_result(sub_out)}


async def docs_researcher_node(state: AgentState) -> AgentState:
    sub_in = {
        "question": effective_question(state),
        "user_role": state["user_role"],
        "company_id": state["company_id"],
    }
    sub_out = await _docs_researcher_compiled().ainvoke(sub_in)
    return {"docs_result": _project_docs_result(sub_out)}
