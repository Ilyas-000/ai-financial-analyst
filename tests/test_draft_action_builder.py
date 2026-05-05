"""Unit tests for app.tools.draft_action_builder."""

from datetime import date
from decimal import Decimal

import pytest

from app.tools.draft_action_builder import (
    DraftAction,
    DraftActionError,
    build_draft_action,
    build_export_report,
    build_highlight_discrepancy,
    build_open_ticket,
    build_prepare_act,
)

# ---------------------------- export_report ----------------------------


def test_export_report_happy_path():
    action = build_export_report(
        report_type="quarterly_summary",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 3, 31),
        currency="rub",
        fmt="xlsx",
    )
    assert isinstance(action, DraftAction)
    assert action.kind == "export_report"
    assert action.requires_confirmation is True
    assert action.payload == {
        "report_type": "quarterly_summary",
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
        "currency": "RUB",
        "format": "xlsx",
    }
    assert "quarterly_summary" in action.title


def test_export_report_inverted_period_rejected():
    with pytest.raises(DraftActionError, match="after"):
        build_export_report(
            report_type="quarterly_summary",
            period_start=date(2025, 4, 1),
            period_end=date(2025, 3, 31),
            currency="RUB",
            fmt="csv",
        )


def test_export_report_unknown_format_rejected():
    with pytest.raises(DraftActionError, match="format"):
        build_export_report(
            report_type="x",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            currency="USD",
            fmt="docx",
        )


def test_export_report_invalid_currency_rejected():
    with pytest.raises(DraftActionError, match="ISO 4217"):
        build_export_report(
            report_type="x",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            currency="rouble",
            fmt="csv",
        )


def test_export_report_empty_report_type_rejected():
    with pytest.raises(DraftActionError, match="report_type"):
        build_export_report(
            report_type="   ",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            currency="USD",
            fmt="csv",
        )


# ---------------------------- open_ticket ----------------------------


def test_open_ticket_happy_path():
    action = build_open_ticket(
        topic="расхождение комиссии",
        severity="high",
        summary="фактическая комиссия не совпадает с тарифом",
        related_ids={"transactions": [101, 102]},
    )
    assert action.kind == "open_ticket"
    assert action.payload["severity"] == "high"
    assert action.payload["related_ids"] == {"transactions": [101, 102]}
    assert action.requires_confirmation is True


def test_open_ticket_no_related_ids_defaults_to_empty():
    action = build_open_ticket(topic="t", severity="low", summary="s")
    assert action.payload["related_ids"] == {}


def test_open_ticket_bad_severity_rejected():
    with pytest.raises(DraftActionError, match="severity"):
        build_open_ticket(topic="t", severity="urgent", summary="s")


def test_open_ticket_empty_topic_rejected():
    with pytest.raises(DraftActionError, match="topic"):
        build_open_ticket(topic="", severity="low", summary="s")


def test_open_ticket_related_ids_with_bool_rejected():
    with pytest.raises(DraftActionError):
        build_open_ticket(
            topic="t",
            severity="low",
            summary="s",
            related_ids={"transactions": [True]},  # type: ignore[list-item]
        )


# ---------------------------- prepare_act ----------------------------


def test_prepare_act_happy_path():
    action = build_prepare_act(
        counterparty_name="ООО «Подрядчик»",
        period_start=date(2025, 4, 1),
        period_end=date(2025, 4, 30),
        payout_ids=[5, 6, 7],
        counterparty_inn="7700000123",
    )
    assert action.kind == "prepare_act"
    assert action.payload["payout_ids"] == [5, 6, 7]
    assert action.payload["counterparty_inn"] == "7700000123"
    assert "Подрядчик" in action.title


def test_prepare_act_optional_inn_defaults_to_none():
    action = build_prepare_act(
        counterparty_name="X",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 1, 31),
        payout_ids=[1],
    )
    assert action.payload["counterparty_inn"] is None


def test_prepare_act_empty_payout_ids_rejected():
    with pytest.raises(DraftActionError, match="payout_ids"):
        build_prepare_act(
            counterparty_name="X",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            payout_ids=[],
        )


def test_prepare_act_period_inverted_rejected():
    with pytest.raises(DraftActionError, match="after"):
        build_prepare_act(
            counterparty_name="X",
            period_start=date(2025, 5, 1),
            period_end=date(2025, 4, 1),
            payout_ids=[1],
        )


# ---------------------------- highlight_discrepancy ----------------------------


def test_highlight_discrepancy_with_decimals():
    action = build_highlight_discrepancy(
        dimension="fee_amount",
        expected=Decimal("100.50"),
        actual=Decimal("105.00"),
        refs={"payouts": [42]},
    )
    assert action.kind == "highlight_discrepancy"
    assert action.payload["expected"] == "100.50"
    assert action.payload["actual"] == "105.00"
    assert action.payload["refs"] == {"payouts": [42]}


def test_highlight_discrepancy_with_floats_uses_repr():
    action = build_highlight_discrepancy(
        dimension="ratio",
        expected=0.5,
        actual=0.55,
    )
    assert action.payload["expected"] == "0.5"
    assert action.payload["actual"] == "0.55"


def test_highlight_discrepancy_string_passthrough():
    action = build_highlight_discrepancy(
        dimension="limit_breach",
        expected="500 000 ₽",
        actual="612 000 ₽",
    )
    assert action.payload["expected"] == "500 000 ₽"


def test_highlight_discrepancy_bool_rejected():
    with pytest.raises(DraftActionError, match="bool"):
        build_highlight_discrepancy(dimension="x", expected=True, actual=False)


def test_highlight_discrepancy_empty_dimension_rejected():
    with pytest.raises(DraftActionError, match="dimension"):
        build_highlight_discrepancy(dimension="", expected=1, actual=2)


# ---------------------------- dispatcher ----------------------------


def test_build_draft_action_dispatches_by_kind():
    action = build_draft_action(
        "open_ticket",
        topic="t",
        severity="low",
        summary="s",
    )
    assert action.kind == "open_ticket"


def test_build_draft_action_unknown_kind_rejected():
    with pytest.raises(DraftActionError, match="unknown draft action kind"):
        build_draft_action("delete_account", topic="x")


def test_as_payload_matches_state_contract():
    action = build_export_report(
        report_type="x",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 1, 31),
        currency="USD",
        fmt="pdf",
    )
    payload = action.as_payload()
    assert set(payload.keys()) == {"kind", "title", "payload", "requires_confirmation"}
    assert payload["requires_confirmation"] is True
    # mutating the snapshot must not affect the original action
    payload["payload"]["currency"] = "EUR"
    assert action.payload["currency"] == "USD"


def test_requires_confirmation_is_invariant_true():
    actions = [
        build_export_report(
            report_type="x",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            currency="RUB",
            fmt="csv",
        ),
        build_open_ticket(topic="t", severity="medium", summary="s"),
        build_prepare_act(
            counterparty_name="x",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            payout_ids=[1],
        ),
        build_highlight_discrepancy(dimension="x", expected=1, actual=2),
    ]
    assert all(a.requires_confirmation is True for a in actions)
