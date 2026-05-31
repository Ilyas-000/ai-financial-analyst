"""Clarify node — asks one short Russian clarifying question.

Used when Supervisor routes to ``clarify`` (ambiguous question, missing
required scope) or when the routing parser falls back. The output replaces
``final_answer``; the chat client / UI shows it as the assistant turn and the
user replies with a more specific question on the next turn.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.llm import FINAL_ANSWER_TAG, invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState, effective_question
from app.rag.tenants import tenant_slug_for

_PROMPT_FILE = "clarify.txt"

# Russian titles for a draft action that arrived together with a clarify route.
_PRE_ACTION_TITLES = {
    "export_report": "Сформировать отчёт",
    "open_ticket": "Открыть тикет",
    "prepare_act": "Подготовить акт",
    "highlight_discrepancy": "Отметить расхождение",
}


def _pre_action(state: AgentState) -> dict[str, Any] | None:
    """Preserve an action intent that arrived on the clarify path (TD-10).

    Supervisor may emit ``suggest_action_kind`` together with ``route=clarify``
    when the user clearly wants an action but the request is under-specified.
    Finalize — the usual action builder — never runs on the clarify branch, so
    without this the intent is silently dropped and the UI shows nothing. We
    emit a lightweight draft in a ``needs_clarification`` state instead: the UI
    can render "draft pending details" and the user's intent survives the turn.
    The full payload is intentionally not built — we lack the missing scope by
    definition (that's why we're clarifying).
    """
    kind = state.get("suggest_action_kind")
    if not kind:
        return None
    question = effective_question(state).strip()
    title = _PRE_ACTION_TITLES.get(kind, "Черновик действия")
    return {
        "kind": kind,
        "title": f"{title} (требует уточнения)",
        "payload": {"question_excerpt": question[:120]} if question else {},
        "requires_confirmation": True,
        "state": "needs_clarification",
    }


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
        "suggested_action": _pre_action(state),
        "messages": [AIMessage(content=answer)],
    }
