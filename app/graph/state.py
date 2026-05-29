"""Parent graph state (design.md §4, I-06–I-08).

* ``route="both"`` fans out to both specialists in parallel (I-08); Finalize joins.
* ``messages`` accumulates across turns (``add_messages``) and is persisted via
  ``PostgresSaver``; everything downstream reads ``condensed_question`` (the
  standalone rewrite) rather than raw ``question`` so follow-ups resolve.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

UserRole = Literal["cfo", "finance_manager", "accountant", "auditor"]
Route = Literal["sql_analyst", "docs_researcher", "both", "direct_answer", "clarify"]
SuggestActionKind = Literal["export_report", "open_ticket", "prepare_act", "highlight_discrepancy"]


class AgentState(TypedDict, total=False):
    # input
    question: str
    user_id: str | None
    user_role: UserRole
    company_id: int
    thread_id: str

    # multi-turn: standalone rewrite of ``question`` produced by condense node
    condensed_question: str | None

    # routing
    route: Route | None
    route_reasoning: str | None
    suggest_action_kind: SuggestActionKind | None

    # specialist outputs (raw subgraph results, kept under namespaced keys
    # so Finalize and observability can inspect them independently)
    sql_result: dict[str, Any] | None
    docs_result: dict[str, Any] | None

    # final
    final_answer: str | None
    sources: list[dict[str, Any]]
    suggested_action: dict[str, Any] | None

    # bookkeeping
    messages: Annotated[list, add_messages]
    tool_calls: list[dict[str, Any]]
    errors: list[str]


def effective_question(state: AgentState) -> str:
    """Question every downstream node should consume.

    Falls back to the raw ``question`` when condense produced nothing —
    happens on the first turn or when the rewrite was skipped.
    """
    condensed = state.get("condensed_question")
    if condensed:
        return condensed
    return state["question"]
