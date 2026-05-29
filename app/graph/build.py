"""Compile the parent agent graph (I-08).

Topology:

    START → condense_question → supervisor → conditional:
        sql_analyst                  ─┐
        docs_researcher              ─┼→ finalize → END
        both → {sql_analyst,         │
                docs_researcher}     ─┘
        direct_answer  → END
        clarify        → END

``condense_question`` rewrites a follow-up into a standalone question from the
checkpointed history, so supervisor/specialists keep single-shot prompts.

For ``route="both"`` the edge fans out to both specialists in parallel; they
write distinct state keys (``sql_result`` / ``docs_result``), so the
``finalize`` join is safe and uses a writer-LLM to synthesise the answer.

The graph is bound to a ``checkpointer`` (real ``AsyncPostgresSaver`` via the
FastAPI lifespan; tests may pass ``None`` for single-turn behaviour).
"""

from langgraph.graph import END, START, StateGraph

from app.graph.clarify import clarify_node
from app.graph.condense import condense_question_node
from app.graph.direct_answer import direct_answer_node
from app.graph.finalize import finalize_node
from app.graph.specialists import docs_researcher_node, sql_analyst_node
from app.graph.state import AgentState
from app.graph.supervisor import route_from_state, supervisor_node


def build_agent_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("condense_question", condense_question_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_analyst", sql_analyst_node)
    graph.add_node("docs_researcher", docs_researcher_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "condense_question")
    graph.add_edge("condense_question", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_state,
        # ``route_from_state`` may return either a single key or a list of keys
        # (for ``route="both"``). LangGraph then fans out to all listed nodes.
        ["sql_analyst", "docs_researcher", "direct_answer", "clarify"],
    )
    graph.add_edge("sql_analyst", "finalize")
    graph.add_edge("docs_researcher", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)
