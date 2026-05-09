"""Parent graph state.

Contract follows design.md §4 with two adjustments grounded in plan.md / I-06:

* ``route`` excludes ``"both"`` until I-08 (parallel execution lands there).
* ``suggest_action_kind`` is a hint emitted by Supervisor that Finalize uses to
  build a ``suggested_action`` payload via ``draft_action_builder``.

``messages`` is wired in now to keep the contract stable for I-07
(PostgresSaver + multi-turn); I-06 itself does not yet use it.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

UserRole = Literal["cfo", "finance_manager", "accountant", "auditor"]
Route = Literal["sql_analyst", "docs_researcher", "direct_answer", "clarify"]
SuggestActionKind = Literal["export_report", "open_ticket", "prepare_act", "highlight_discrepancy"]


class AgentState(TypedDict, total=False):
    # input
    question: str
    user_id: str | None
    user_role: UserRole
    company_id: int
    thread_id: str

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
