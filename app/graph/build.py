"""Compile the parent agent graph.

Topology (I-08):

    START → condense_question → supervisor → conditional:
        sql_analyst                  ─┐
        docs_researcher              ─┼→ finalize → END
        both → {sql_analyst,         │
                docs_researcher}     ─┘
        direct_answer  → END
        clarify        → END

The ``condense_question`` node rewrites a follow-up question into a
standalone one using the prior turns persisted by the checkpointer, so
supervisor and specialists keep their single-shot prompts unchanged.

For ``route="both"`` the conditional edge fans out into ``sql_analyst`` and
``docs_researcher`` in parallel; both branches write into different state
keys (``sql_result`` / ``docs_result``), so the join at ``finalize`` is safe.
Finalize then uses a writer-LLM to synthesise the combined answer.

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
