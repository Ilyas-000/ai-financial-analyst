"""Clarify node — asks one short Russian clarifying question.

Used when Supervisor routes to ``clarify`` (ambiguous question, missing
required scope) or when the routing parser falls back. The output replaces
``final_answer``; the chat client / UI shows it as the assistant turn and the
user replies with a more specific question on the next turn.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.llm import FINAL_ANSWER_TAG, invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState, effective_question
from app.rag.tenants import tenant_slug_for

_PROMPT_FILE = "clarify.txt"


async def clarify_node(state: AgentState) -> AgentState:
    tenant_slug = await tenant_slug_for(state["company_id"])
    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    # ``reasoning`` is consumed by the SYSTEM section (routing hint), not USER —
    # keep the .format calls aligned with the actual placeholders in the prompt
    # file or LangChain raises KeyError before the LLM is ever reached.
    reasoning = state.get("route_reasoning") or "ambiguous question"
    system_prompt = system_template.format(
        user_role=state["user_role"],
        tenant_slug=tenant_slug,
        reasoning=reasoning,
    )
    user_prompt = user_template.format(question=effective_question(state))

    # ``final_answer`` tag — see ``app/graph/direct_answer.py``.
    llm = make_llm("writer", tags=[FINAL_ANSWER_TAG])
    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    answer = str(response.content).strip()
    return {
        "final_answer": answer,
        "sources": [],
        "suggested_action": None,
        "messages": [AIMessage(content=answer)],
    }
