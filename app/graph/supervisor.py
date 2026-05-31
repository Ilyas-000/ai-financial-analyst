"""Supervisor routing node.

Emits ``{route, route_reasoning, suggest_action_kind}`` into ``AgentState``.

Following design.md §4 we wanted Pydantic ``with_structured_output``, but on
the 3B model used locally that path was unreliable in I-04. Instead we ask
the LLM for a strict JSON object and parse it manually with one bounded
re-prompt on failure. Hard fallback: ``route="clarify"`` with a reasoning
hint that the routing decision could not be parsed.
"""

import json
import re
from typing import Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.graph.llm import invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt, strip_code_fences
from app.graph.state import AgentState, effective_question

logger = structlog.get_logger(__name__)

_PROMPT_FILE = "supervisor.txt"
_FORMAT_REMINDER = (
    "\n\nFORMAT REMINDER: respond with EXACTLY one JSON object on a single line, "
    'with keys "next", "reasoning", "suggest_action_kind". No markdown, no prose.'
)

# Deterministic pre-route patterns (tech_debt TD-07 / TD-09). The local 7B
# Supervisor consistently mis-routes three question shapes even after prompt
# tuning: a scopeless spend request (routes to sql_analyst and fabricates a
# period), a normative role-limit lookup (routes to sql_analyst), and a
# "how-to" procedure (routes to sql_analyst or clarify). These narrow,
# high-precision rules short-circuit those shapes; everything else falls
# through to the LLM. Domain vocabulary, not config — kept as constants.
_HOWTO_MARKERS = (
    "как выгрузить",
    "как оформить",
    "как закрыть",
    "как сделать",
    "как получить",
    "как провести",
    "как согласовать",
    "как заполнить",
    "каков порядок",
    "какова процедура",
    "какая процедура",
    "куда эскалир",
)
_ROLE_WORDS = (
    "топ-менедж",
    "руковод",
    "сотрудник",
    "менеджер",
    "бухгалтер",
    "директор",
    "аудитор",
    "персонал",
    "роли",
    "роль",
)
# Signals that a limit/tariff question is about OUR rows, not a policy value.
_OURDATA_SIGNALS = ("у нас", "наши", "наш ", "в систем", "задан", "настроен", "сколько")
_SPEND_STEMS = ("потрат", "израсход", "расход", "заплат")
# A rules/policy ask alongside a spend verb makes the question a likely `both`
# or `docs` case (e.g. "тарифная политика и сколько мы заплатили ..."). Such
# questions must reach the LLM, never be forced to clarify as scopeless spend.
_RULES_SIGNALS = (
    "регламент",
    "политик",
    "тариф",
    "шаблон",
    "классифи",
    "процедур",
    "норматив",
    "правил",
    "эскалац",
)
# A comparison request ("сравни наши расходы с ...") always names a second
# operand, so it is never the vague scopeless figure the clarify rule targets.
# Crucially, some comparisons are cross-tenant attack shapes that must reach
# the LLM → sql_analyst → tenant guard; the pre-route must not short-circuit
# them to clarify and defang the security path.
_COMPARISON_MARKERS = ("сравн",)
_SCOPE_TOKENS = (
    "квартал",
    "месяц",
    "год",
    "недел",
    "период",
    "сегодня",
    "вчера",
    "за всё время",
    "за все время",
    "командиров",
    "отел",
    "такси",
    "реклам",
    "категори",
    "по карт",
    "подрядчик",
    "в отдел",
)
_YEAR_OR_QUARTER = re.compile(r"\b(?:19|20)\d{2}\b|\bq[1-4]\b|\bкв[\s.]*[1-4]\b")


class _Route(BaseModel):
    next: Literal["sql_analyst", "docs_researcher", "both", "direct_answer", "clarify"]
    reasoning: str = Field(min_length=1, max_length=500)
    suggest_action_kind: (
        Literal["export_report", "open_ticket", "prepare_act", "highlight_discrepancy"] | None
    ) = None


def _has_scope(question: str) -> bool:
    if _YEAR_OR_QUARTER.search(question):
        return True
    return any(token in question for token in _SCOPE_TOKENS)


def pre_route(question: str) -> _Route | None:
    """High-precision deterministic routing for shapes the 7B mis-routes.

    Returns a forced route or ``None`` to defer to the Supervisor LLM.
    """
    text = question.strip().lower()
    if any(marker in text for marker in _HOWTO_MARKERS):
        return _Route(
            next="docs_researcher",
            reasoning="how-to / procedure question routed by deterministic pre-route",
        )
    if (
        "лимит" in text
        and any(role in text for role in _ROLE_WORDS)
        and not any(signal in text for signal in _OURDATA_SIGNALS)
    ):
        return _Route(
            next="docs_researcher",
            reasoning="normative role limit is a policy value, deterministic pre-route",
        )
    if (
        any(stem in text for stem in _SPEND_STEMS)
        and not _has_scope(text)
        and not any(signal in text for signal in _RULES_SIGNALS)
        and not any(marker in text for marker in _COMPARISON_MARKERS)
    ):
        return _Route(
            next="clarify",
            reasoning="scopeless spend request with no period or dimension, deterministic pre-route",
        )
    return None


def _extract_json_object(raw: str) -> str | None:
    cleaned = strip_code_fences(raw)
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
    question = effective_question(state)

    forced = pre_route(question)
    if forced is not None:
        logger.info("supervisor_pre_route_hit", route=forced.next)
        structlog.contextvars.bind_contextvars(route=forced.next)
        return {
            "route": forced.next,
            "route_reasoning": forced.reasoning,
            "suggest_action_kind": forced.suggest_action_kind,
        }

    system_template, user_template = load_two_section_prompt(_PROMPT_FILE)
    system_prompt = system_template.format(
        user_role=state["user_role"], company_id=state["company_id"]
    )
    user_prompt = user_template.format(question=question)

    llm = make_llm("supervisor")

    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    route = parse_route(str(response.content))

    if route is None:
        logger.warning("supervisor_parse_failed_retrying")
        retry_user = user_prompt + _FORMAT_REMINDER
        response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(retry_user)])
        route = parse_route(str(response.content))

    if route is None:
        logger.warning("supervisor_routing_fallback_clarify")
        structlog.contextvars.bind_contextvars(route="clarify")
        return {
            "route": "clarify",
            "route_reasoning": "failed to parse routing decision",
            "suggest_action_kind": None,
        }

    structlog.contextvars.bind_contextvars(route=route.next)
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
