"""Clarify node — asks one short Russian clarifying question.

Used when Supervisor routes to ``clarify`` (ambiguous question, missing
required scope) or when the routing parser falls back. The output replaces
``final_answer``; the chat client / UI shows it as the assistant turn and the
user replies with a more specific question on the next turn.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState
from app.rag.tenants import tenant_slug_for

_PROMPT_FILE = "clarify.txt"


async def clarify_node(state: AgentState) -> AgentState:
    tenant_slug = await tenant_slug_for(state["company_id"])
    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    system_prompt = system_template.format(user_role=state["user_role"], tenant_slug=tenant_slug)
    user_prompt = user_template.format(
        question=state["question"],
        reasoning=state.get("route_reasoning") or "ambiguous question",
    )

    llm = make_llm("writer")
    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    answer = str(response.content).strip()
    return {
        "final_answer": answer,
        "sources": [],
        "suggested_action": None,
    }
