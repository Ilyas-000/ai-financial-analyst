"""Condense the latest user question into a standalone form (I-07).

Supervisor and specialists only see ``effective_question(state)``, so this
first node rewrites follow-ups against history to be self-contained
("а за Q2?" → "Сколько потратили на командировки в Q2 2025?"). First turn
skips the rewrite. It also appends the new ``HumanMessage`` to ``messages``
(terminal nodes append the matching ``AIMessage``) to keep the checkpointed
conversation log complete.
"""

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt, strip_code_fences
from app.graph.state import AgentState

logger = structlog.get_logger(__name__)

_PROMPT_FILE = "condense_question.txt"

# Cap conversation context fed to the rewriter. The whole point is to inject
# the missing scope from recent turns; older turns rarely add signal and would
# just inflate the prompt.
_HISTORY_TURN_LIMIT = 6
_MESSAGE_CHAR_CAP = 600
_REWRITE_CHAR_CAP = 500


def _has_prior_assistant_turn(messages: list) -> bool:
    return any(isinstance(message, AIMessage) for message in messages)


def _truncate(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "…"


def _format_history(messages: list) -> str:
    recent = messages[-_HISTORY_TURN_LIMIT:]
    lines: list[str] = []
    for message in recent:
        if isinstance(message, HumanMessage):
            label = "User"
        elif isinstance(message, AIMessage):
            label = "Assistant"
        else:
            continue
        content = _truncate(str(message.content), _MESSAGE_CHAR_CAP)
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


_QUOTE_PAIRS = {'"': '"', "'": "'", "«": "»", "“": "”"}


def _strip_wrappers(text: str) -> str:
    text = strip_code_fences(text)
    if len(text) >= 2:
        closer = _QUOTE_PAIRS.get(text[0])
        if closer is not None and text[-1] == closer:
            text = text[1:-1].strip()
    return text


def _per_turn_reset() -> dict:
    """Wipe single-turn fields persisted by the prior checkpoint.

    ``messages`` is the only field that should accumulate across turns
    (multi-turn log). Everything else — routing decision, specialist outputs,
    finalised answer, sources, suggested_action, errors, tool_calls — is per
    turn; without an explicit reset a stale ``docs_result`` from turn N-1
    can leak into Finalize on turn N if the route changed.
    """
    return {
        "route": None,
        "route_reasoning": None,
        "suggest_action_kind": None,
        "sql_result": None,
        "docs_result": None,
        "final_answer": None,
        "sources": [],
        "suggested_action": None,
        "errors": [],
        "tool_calls": [],
    }


async def condense_question_node(state: AgentState) -> AgentState:
    question = state["question"]
    prior_messages = state.get("messages") or []
    new_user_message = HumanMessage(content=question)

    if not _has_prior_assistant_turn(prior_messages):
        # First turn (or recovery): no context to fold in.
        return {
            **_per_turn_reset(),
            "condensed_question": question,
            "messages": [new_user_message],
        }

    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    user_prompt = user_template.format(
        history=_format_history(prior_messages),
        question=question,
    )

    llm = make_llm("supervisor")
    response = await invoke_llm(llm, [SystemMessage(system_template), HumanMessage(user_prompt)])
    rewritten = _strip_wrappers(str(response.content))
    if not rewritten:
        logger.warning("condense_empty_rewrite_fallback")
        rewritten = question
    elif len(rewritten) > _REWRITE_CHAR_CAP:
        logger.info("condense_rewrite_truncated", length=len(rewritten))
        rewritten = _truncate(rewritten, _REWRITE_CHAR_CAP)

    return {
        **_per_turn_reset(),
        "condensed_question": rewritten,
        "messages": [new_user_message],
    }
