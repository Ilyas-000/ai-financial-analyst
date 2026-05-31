"""Unit tests for the Supervisor JSON parser and the conditional-edge function.

We only test the pure helpers (``parse_route``, ``route_from_state``). The node
itself is covered end-to-end via ``scripts/chat.py`` against a live Ollama
instance.
"""

import pytest

from app.graph.supervisor import parse_route, pre_route, route_from_state


def test_parse_clean_json():
    raw = '{"next": "sql_analyst", "reasoning": "needs aggregation", "suggest_action_kind": null}'
    route = parse_route(raw)
    assert route is not None
    assert route.next == "sql_analyst"
    assert route.reasoning == "needs aggregation"
    assert route.suggest_action_kind is None


def test_parse_with_action_hint():
    raw = '{"next": "sql_analyst", "reasoning": "report", "suggest_action_kind": "export_report"}'
    route = parse_route(raw)
    assert route is not None
    assert route.suggest_action_kind == "export_report"


def test_parse_strips_code_fences():
    raw = (
        "```json\n"
        '{"next": "docs_researcher", "reasoning": "policy lookup",'
        ' "suggest_action_kind": null}\n'
        "```"
    )
    route = parse_route(raw)
    assert route is not None
    assert route.next == "docs_researcher"


def test_parse_extracts_object_from_prose():
    raw = (
        "Here is the routing decision: "
        '{"next": "clarify", "reasoning": "missing period", "suggest_action_kind": null} '
        "— hope this helps."
    )
    route = parse_route(raw)
    assert route is not None
    assert route.next == "clarify"


def test_parse_invalid_route_value_returns_none():
    raw = '{"next": "wizardry", "reasoning": "x", "suggest_action_kind": null}'
    assert parse_route(raw) is None


def test_parse_invalid_action_kind_returns_none():
    raw = '{"next": "sql_analyst", "reasoning": "x", "suggest_action_kind": "send_email"}'
    assert parse_route(raw) is None


def test_parse_missing_keys_returns_none():
    raw = '{"next": "sql_analyst"}'
    assert parse_route(raw) is None


def test_parse_empty_reasoning_returns_none():
    raw = '{"next": "sql_analyst", "reasoning": "", "suggest_action_kind": null}'
    assert parse_route(raw) is None


def test_parse_non_object_returns_none():
    assert parse_route("[1, 2, 3]") is None
    assert parse_route("just prose without any json") is None
    assert parse_route("") is None


def test_parse_malformed_json_returns_none():
    raw = '{"next": "sql_analyst", "reasoning": "broken"'
    assert parse_route(raw) is None


def test_parse_both_route():
    raw = (
        '{"next": "both", "reasoning": "fact vs regulation comparison",'
        ' "suggest_action_kind": null}'
    )
    route = parse_route(raw)
    assert route is not None
    assert route.next == "both"


def test_route_from_state_fans_out_for_both():
    out = route_from_state({"route": "both"})
    assert out == ["sql_analyst", "docs_researcher"]


def test_route_from_state_returns_single_key_for_other_routes():
    assert route_from_state({"route": "sql_analyst"}) == "sql_analyst"
    assert route_from_state({"route": "docs_researcher"}) == "docs_researcher"
    assert route_from_state({"route": "direct_answer"}) == "direct_answer"
    assert route_from_state({"route": "clarify"}) == "clarify"


def test_route_from_state_defaults_to_clarify_when_missing():
    assert route_from_state({}) == "clarify"
    assert route_from_state({"route": None}) == "clarify"


# --- pre_route deterministic guard -----------------------------------------


def test_pre_route_howto_goes_to_docs():
    forced = pre_route("Как выгрузить закрывающие документы за период?")
    assert forced is not None
    assert forced.next == "docs_researcher"


def test_pre_route_howto_procedure_phrasings():
    for q in (
        "Как оформить командировку?",
        "Каков порядок согласования трат?",
        "Какова процедура закрытия месяца?",
        "Куда эскалировать инцидент?",
    ):
        forced = pre_route(q)
        assert forced is not None, q
        assert forced.next == "docs_researcher", q


def test_pre_route_role_limit_goes_to_docs():
    forced = pre_route("Какой месячный лимит у топ-менеджеров на корпоративные карты?")
    assert forced is not None
    assert forced.next == "docs_researcher"


def test_pre_route_role_limit_for_other_roles():
    for q in (
        "Какой лимит у руководителей?",
        "Какой лимит на корпкарту у бухгалтера?",
        "Какой лимит у сотрудника на такси?",
    ):
        forced = pre_route(q)
        assert forced is not None, q
        assert forced.next == "docs_researcher", q


def test_pre_route_our_data_limit_defers_to_llm():
    # "лимиты заданы в системе для нашей компании" is our-data → must NOT be
    # hijacked to docs by the role-limit rule (ourdata signal blocks it).
    assert pre_route("Какие лимиты заданы в системе для нашей компании?") is None


def test_pre_route_limit_without_role_defers():
    # No role word → role-limit rule must not fire.
    assert pre_route("Какой лимит на эту операцию?") is None


def test_pre_route_scopeless_spend_goes_to_clarify():
    forced = pre_route("Сколько мы потратили?")
    assert forced is not None
    assert forced.next == "clarify"


def test_pre_route_scopeless_spend_stems():
    for q in (
        "Сколько мы израсходовали?",
        "Каковы наши расходы?",
        "Сколько заплатили?",
    ):
        forced = pre_route(q)
        assert forced is not None, q
        assert forced.next == "clarify", q


def test_pre_route_spend_with_period_defers():
    # A scope (period) is present → defer to the LLM, do not force clarify.
    assert pre_route("Сколько потратили на командировки в Q1?") is None
    assert pre_route("Сколько мы потратили за прошлый месяц?") is None
    assert pre_route("Сколько потратили на рекламу в 2025?") is None


def test_pre_route_spend_with_dimension_defers():
    # A category/dimension counts as scope even without an explicit period.
    assert pre_route("Сколько мы потратили на такси?") is None
    assert pre_route("Сколько потратили по карте подрядчику?") is None


def test_pre_route_plain_count_defers():
    # A count of our records is sql_analyst territory — pre_route must defer.
    assert pre_route("Сколько у нас транзакций?") is None
    assert pre_route("Покажи наши тарифы.") is None


def test_pre_route_does_not_hijack_both_shape():
    # Data + rules dual ask must reach the LLM (it has a period scope).
    assert (
        pre_route(
            "Сравни наши расходы по корпкартам за апрель с месячным лимитом из регламента."
        )
        is None
    )


def test_pre_route_scopeless_spend_with_rules_signal_defers():
    # A spend verb next to a policy/rules ask is a likely `both`/`docs` case,
    # not a scopeless figure — must defer to the LLM even without a scope token.
    assert pre_route("Какова тарифная политика и сколько мы заплатили комиссий за payout'ы?") is None
    assert pre_route("Сколько мы потратили и что говорит регламент?") is None


def test_pre_route_comparison_spend_defers():
    # A "сравни ..." comparison names a second operand — not the vague scopeless
    # figure the clarify rule targets. Cross-tenant compares MUST reach the LLM
    # → sql_analyst → tenant guard, not be short-circuited to clarify.
    assert pre_route("Сравни наши расходы с расходами ACME LLC") is None
    assert pre_route("Сравнить наши траты с прошлым годом") is None


def test_pre_route_returns_none_for_greeting():
    assert pre_route("Привет, как пользоваться этим помощником?") is None
