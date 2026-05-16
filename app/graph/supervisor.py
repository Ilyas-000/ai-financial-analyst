"""Supervisor routing node.

Emits ``{route, route_reasoning, suggest_action_kind}`` into ``AgentState``.

Following design.md §4 we wanted Pydantic ``with_structured_output``, but on
the 3B model used locally that path was unreliable in I-04. Instead we ask
the LLM for a strict JSON object and parse it manually with one bounded
re-prompt on failure. Hard fallback: ``route="clarify"`` with a reasoning
hint that the routing decision could not be parsed.
"""

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.graph.state import AgentState, effective_question

logger = logging.getLogger(__name__)

_PROMPT_FILE = "supervisor.txt"
_FORMAT_REMINDER = (
    "\n\nFORMAT REMINDER: respond with EXACTLY one JSON object on a single line, "
    'with keys "next", "reasoning", "suggest_action_kind". No markdown, no prose.'
)


class _Route(BaseModel):
    next: Literal["sql_analyst", "docs_researcher", "both", "direct_answer", "clarify"]
    reasoning: str = Field(min_length=1, max_length=500)
    suggest_action_kind: (
        Literal["export_report", "open_ticket", "prepare_act", "highlight_discrepancy"] | None
    ) = None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def _extract_json_object(raw: str) -> str | None:
    cleaned = _strip_fences(raw)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return cleaned[start : end + 1]


def parse_route(raw: str) -> _Route | None:
    candidate = _extract_json_object(raw)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return _Route.model_validate(data)
    except ValidationError:
        return None


async def supervisor_node(state: AgentState) -> AgentState:
    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    system_prompt = system_template.format(
        user_role=state["user_role"], company_id=state["company_id"]
    )
    user_prompt = user_template.format(question=effective_question(state))

    llm = make_llm("supervisor")

    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    route = parse_route(str(response.content))

    if route is None:
        logger.warning("supervisor returned non-parseable JSON, retrying with format reminder")
        retry_user = user_prompt + _FORMAT_REMINDER
        response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(retry_user)])
        route = parse_route(str(response.content))

    if route is None:
        logger.warning("supervisor routing fallback to clarify")
        return {
            "route": "clarify",
            "route_reasoning": "failed to parse routing decision",
            "suggest_action_kind": None,
        }

    return {
        "route": route.next,
        "route_reasoning": route.reasoning,
        "suggest_action_kind": route.suggest_action_kind,
    }


def route_from_state(state: AgentState) -> str | list[str]:
    """Conditional-edge function — picks the next node(s) based on Supervisor output.

    Returns a list when ``route == "both"`` so LangGraph fans out to both
    specialists in parallel; their results are joined at ``finalize``.
    """
    route = state.get("route") or "clarify"
    if route == "both":
        return ["sql_analyst", "docs_researcher"]
    return route
