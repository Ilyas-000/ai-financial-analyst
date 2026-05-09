"""Direct-answer node — short Russian reply for small talk and capability questions.

The Supervisor routes here when the question needs neither tenant data nor an
internal regulation. We deliberately do not pass through Finalize: the LLM
output is already the final user-facing answer, and there are no sources or
draft actions to attach.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState
from app.rag.tenants import tenant_slug_for

_PROMPT_FILE = "direct_answer.txt"


async def direct_answer_node(state: AgentState) -> AgentState:
    tenant_slug = await tenant_slug_for(state["company_id"])
    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    system_prompt = system_template.format(user_role=state["user_role"], tenant_slug=tenant_slug)
    user_prompt = user_template.format(question=state["question"])

    llm = make_llm("writer")
    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    answer = str(response.content).strip()
    return {
        "final_answer": answer,
        "sources": [],
        "suggested_action": None,
    }
