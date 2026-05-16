"""Compile the parent agent graph.

Topology (I-07):

    START → condense_question → supervisor → conditional:
        sql_analyst    → finalize → END
        docs_researcher → finalize → END
        direct_answer  → END
        clarify        → END

The ``condense_question`` node rewrites a follow-up question into a
standalone one using the prior turns persisted by the checkpointer, so
supervisor and specialists keep their single-shot prompts unchanged.

``route=both`` and parallel fan-out are scoped to I-08.

The compiled graph is bound to a ``checkpointer`` so each run with a given
``thread_id`` resumes the conversation log. Production wires the real
``AsyncPostgresSaver`` through the FastAPI lifespan; tests can pass ``None``
and get the legacy single-turn behaviour.
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
        {
            "sql_analyst": "sql_analyst",
            "docs_researcher": "docs_researcher",
            "direct_answer": "direct_answer",
            "clarify": "clarify",
        },
    )
    graph.add_edge("sql_analyst", "finalize")
    graph.add_edge("docs_researcher", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)
