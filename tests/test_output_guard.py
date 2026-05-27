"""Unit tests for the I-11 output guardrail."""

import pytest

import app.guardrails.output_guard as output_guard_module
from app.guardrails.output_guard import apply_output_guard
from app.guardrails.tenant_index import TenantEntry

# All tests in this module run on the event loop.
pytestmark = pytest.mark.asyncio

_INDEX = [
    TenantEntry(company_id=1, name="ACME LLC", inn="7700000001"),
    TenantEntry(company_id=2, name="Ostrovok-mock", inn="7800000002"),
    TenantEntry(company_id=3, name="CheckScan-mock", inn="5000000003"),
]


@pytest.fixture(autouse=True)
def stub_index(monkeypatch):
    async def _load():
        return _INDEX

    monkeypatch.setattr(output_guard_module, "load_tenant_index", _load)


@pytest.fixture(autouse=True)
def stub_audit(monkeypatch):
    """Capture audit writes so tests assert on them without hitting Postgres."""
    captured: list[dict] = []

    async def _fake_write_audit(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(output_guard_module, "_write_audit", _fake_write_audit)
    return captured


# --- happy paths --------------------------------------------------------------


async def test_pass_through_when_answer_mentions_only_own_tenant():
    result = await apply_output_guard(
        answer="По компании ACME LLC сумма расходов 1 234.56 ₽.",
        sources=[{"type": "sql", "sql": "SELECT 1"}],
        suggested_action=None,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-ok-1",
    )
    assert not result.rejected
    assert result.answer.startswith("По компании ACME LLC")
    assert result.sources == [{"type": "sql", "sql": "SELECT 1"}]
    assert result.foreign_hits == []


async def test_does_not_flag_bare_company_id_numbers():
    """Plain integers equal to a foreign company_id MUST NOT trigger reject —
    financial text is full of small integers and R3 ruled bare IDs out."""
    result = await apply_output_guard(
        answer="Транзакций: 2. Сумма: 1 000 ₽. Карты: 3.",
        sources=[],
        suggested_action=None,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-ok-2",
    )
    assert not result.rejected


# --- cross-tenant reject ------------------------------------------------------


async def test_rejects_when_foreign_tenant_name_present(stub_audit):
    result = await apply_output_guard(
        answer="Сумма расходов Ostrovok-mock составила 500 000 ₽.",
        sources=[{"type": "sql"}],
        suggested_action={"kind": "export_report", "requires_confirmation": True},
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-bad-1",
    )
    assert result.rejected
    assert result.rejection_reason == "cross_tenant_leak"
    assert result.sources == []
    assert result.suggested_action is None
    assert "отклонён" in result.answer.lower() or "безопасности" in result.answer.lower()
    assert result.foreign_hits == ["name:Ostrovok-mock"]
    # audit row recorded with severity=high
    assert len(stub_audit) == 1
    audit = stub_audit[0]
    assert audit["severity"] == "high"
    assert audit["action"] == "output_blocked_cross_tenant"
    assert audit["company_id"] == 1
    assert audit["user_role"] == "finance_manager"


async def test_rejects_when_foreign_inn_present(stub_audit):
    result = await apply_output_guard(
        answer="Контрагент с ИНН 7800000002 получил выплату.",
        sources=[],
        suggested_action=None,
        user_company_id=1,
        user_role="cfo",
        thread_id="t-bad-2",
    )
    assert result.rejected
    assert result.foreign_hits == ["inn:7800000002"]
    assert len(stub_audit) == 1


async def test_case_insensitive_name_match(stub_audit):
    result = await apply_output_guard(
        answer="данные по OSTROVOK-MOCK здесь",
        sources=[],
        suggested_action=None,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-bad-3",
    )
    assert result.rejected
    assert any(h.startswith("name:") for h in result.foreign_hits)


async def test_own_tenant_inn_not_flagged():
    result = await apply_output_guard(
        answer="ИНН вашей компании: 7700000001.",
        sources=[],
        suggested_action=None,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-ok-3",
    )
    assert not result.rejected


# --- masking and action invariant --------------------------------------------


async def test_masks_card_number_in_answer():
    result = await apply_output_guard(
        answer="Списание прошло с карты 4111 1111 1111 1111.",
        sources=[],
        suggested_action=None,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-mask-1",
    )
    assert not result.rejected
    assert "**** 1111" in result.answer
    assert "4111 1111 1111 1111" not in result.answer
    assert result.masked_cards == 1


async def test_drops_action_without_requires_confirmation():
    bad_action = {
        "kind": "open_ticket",
        "title": "x",
        "payload": {},
        "requires_confirmation": False,
    }
    result = await apply_output_guard(
        answer="ок",
        sources=[],
        suggested_action=bad_action,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-act-1",
    )
    assert result.suggested_action is None
    assert result.dropped_action is True


async def test_keeps_action_with_requires_confirmation():
    good_action = {
        "kind": "open_ticket",
        "title": "x",
        "payload": {},
        "requires_confirmation": True,
    }
    result = await apply_output_guard(
        answer="ок",
        sources=[],
        suggested_action=good_action,
        user_company_id=1,
        user_role="finance_manager",
        thread_id="t-act-2",
    )
    assert result.suggested_action == good_action
    assert result.dropped_action is False
