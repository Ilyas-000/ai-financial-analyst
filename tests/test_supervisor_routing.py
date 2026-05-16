"""Unit tests for the Supervisor JSON parser and the conditional-edge function.

We only test the pure helpers (``parse_route``, ``route_from_state``). The node
itself is covered end-to-end via ``scripts/chat.py`` against a live Ollama
instance.
"""

from app.graph.supervisor import parse_route, route_from_state


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
