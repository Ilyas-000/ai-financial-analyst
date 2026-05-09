"""Compile the parent agent graph and expose a cached factory.

Topology (I-06):

    START → supervisor → conditional:
        sql_analyst    → finalize → END
        docs_researcher → finalize → END
        direct_answer  → END
        clarify        → END

``route=both`` and parallel fan-out are scoped to I-08; this build keeps
single-source paths only.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.clarify import clarify_node
from app.graph.direct_answer import direct_answer_node
from app.graph.finalize import finalize_node
from app.graph.specialists import docs_researcher_node, sql_analyst_node
from app.graph.state import AgentState
from app.graph.supervisor import route_from_state, supervisor_node


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_analyst", sql_analyst_node)
    graph.add_node("docs_researcher", docs_researcher_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "supervisor")
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

    return graph.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    return build_agent_graph()
