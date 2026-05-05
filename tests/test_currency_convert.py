"""Unit tests for app.tools.currency_convert.

Network and DB are mocked: orchestration is tested by patching the four
internal helpers (_query_exact / _query_last_known / _fetch_upstream /
_persist_rates) and replacing the sessionmaker with a no-op fake. The
upstream fetch path itself is tested separately via httpx.MockTransport.
"""

from datetime import date
from decimal import Decimal

import httpx
import pytest

import app.tools.currency_convert as ccm
from app.tools.currency_convert import (
    ConversionResult,
    CurrencyConvertError,
    _RateHit,
    currency_convert,
)


class _FakeSessionTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def begin(self):
        return _FakeSessionTx()


def _fake_sessionmaker():
    return lambda: _FakeSession()


@pytest.fixture
def patch_session(monkeypatch):
    monkeypatch.setattr(ccm, "get_sessionmaker", _fake_sessionmaker)


@pytest.fixture
def stub_helpers(monkeypatch):
    """Replace internal helpers with controllable stubs.

    Tests register hits via ``stub.exact[(base, quote, day)] = _RateHit(...)`` /
    ``stub.last[(base, quote)] = _RateHit(...)`` / ``stub.upstream = (date, rates, source)``.
    """

    class _Stub:
        def __init__(self):
            self.exact: dict[tuple[str, str, date], _RateHit] = {}
            self.last: dict[tuple[str, str], _RateHit] = {}
            self.upstream: tuple[date, dict[str, Decimal], str] | None = None
            self.fetch_calls: int = 0
            self.persist_calls: list[tuple[str, date, dict[str, Decimal]]] = []

    stub = _Stub()

    async def fake_query_exact(_session, base, quote, target_date):
        return stub.exact.get((base, quote, target_date))

    async def fake_query_last_known(_session, base, quote, target_date):
        hit = stub.last.get((base, quote))
        if hit is None or hit.date_used > target_date:
            return None
        return hit

    async def fake_fetch_upstream(_base):
        stub.fetch_calls += 1
        return stub.upstream

    async def fake_persist_rates(_session, base, fetched_date, rates, _source):
        stub.persist_calls.append((base, fetched_date, dict(rates)))
        # When upstream lands rates, expose them as exact hits for re-query.
        for quote, rate in rates.items():
            stub.exact[(base, quote, fetched_date)] = _RateHit(
                rate=rate, date_used=fetched_date, source=_source
            )

    monkeypatch.setattr(ccm, "_query_exact", fake_query_exact)
    monkeypatch.setattr(ccm, "_query_last_known", fake_query_last_known)
    monkeypatch.setattr(ccm, "_fetch_upstream", fake_fetch_upstream)
    monkeypatch.setattr(ccm, "_persist_rates", fake_persist_rates)
    return stub


# ---------------------------- input validation ----------------------------


async def test_identity_skips_session_and_network(monkeypatch):
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("sessionmaker must not be touched for identity")

    monkeypatch.setattr(ccm, "get_sessionmaker", boom)
    result = await currency_convert(Decimal("100"), "USD", "USD")
    assert called is False
    assert isinstance(result, ConversionResult)
    assert result.amount == Decimal("100")
    assert result.rate == Decimal("1")
    assert result.source == "identity"
    assert result.stale is False


async def test_invalid_from_code_rejected():
    with pytest.raises(CurrencyConvertError, match="from_ccy"):
        await currency_convert(Decimal("1"), "USDD", "RUB")


async def test_invalid_to_code_rejected():
    with pytest.raises(CurrencyConvertError, match="to_ccy"):
        await currency_convert(Decimal("1"), "USD", "ru")


async def test_lowercase_codes_normalized(patch_session, stub_helpers):
    today = date.today()
    stub_helpers.exact[("USD", "RUB", today)] = _RateHit(
        rate=Decimal("90"), date_used=today, source="seed"
    )
    result = await currency_convert(Decimal("10"), "usd", "rub")
    assert result.from_ccy == "USD"
    assert result.to_ccy == "RUB"
    assert result.amount == Decimal("900")


async def test_amount_string_coerced(patch_session, stub_helpers):
    today = date.today()
    stub_helpers.exact[("USD", "RUB", today)] = _RateHit(
        rate=Decimal("90"), date_used=today, source="seed"
    )
    result = await currency_convert("12.5", "USD", "RUB")
    assert result.amount == Decimal("1125.0")


async def test_amount_bool_rejected():
    with pytest.raises(CurrencyConvertError, match="bool"):
        await currency_convert(True, "USD", "RUB")  # type: ignore[arg-type]


async def test_amount_garbage_rejected():
    with pytest.raises(CurrencyConvertError, match="numeric"):
        await currency_convert("not-a-number", "USD", "RUB")


# ---------------------------- resolution paths ----------------------------


async def test_exact_pair_hit_skips_upstream(patch_session, stub_helpers):
    today = date.today()
    stub_helpers.exact[("USD", "RUB", today)] = _RateHit(
        rate=Decimal("90.5"), date_used=today, source="seed"
    )
    result = await currency_convert(Decimal("100"), "USD", "RUB")
    assert stub_helpers.fetch_calls == 0
    assert result.rate == Decimal("90.5")
    assert result.amount == Decimal("9050.0")
    assert result.stale is False
    assert result.source == "seed"
    assert result.date_used == today


async def test_inverted_pair_hit_uses_reciprocal(patch_session, stub_helpers):
    today = date.today()
    stub_helpers.exact[("RUB", "USD", today)] = _RateHit(
        rate=Decimal("0.01"), date_used=today, source="seed"
    )
    result = await currency_convert(Decimal("100"), "USD", "RUB")
    assert stub_helpers.fetch_calls == 0
    assert result.rate == Decimal("100")
    assert result.amount == Decimal("10000")
    assert result.stale is False


async def test_today_miss_triggers_upstream_then_resolves(patch_session, stub_helpers):
    today = date.today()
    stub_helpers.upstream = (
        today,
        {"RUB": Decimal("91"), "EUR": Decimal("0.92")},
        "open.er-api.com",
    )
    result = await currency_convert(Decimal("10"), "USD", "RUB")
    assert stub_helpers.fetch_calls == 1
    assert stub_helpers.persist_calls
    persisted_base, persisted_date, persisted_rates = stub_helpers.persist_calls[0]
    assert persisted_base == "USD"
    assert persisted_date == today
    assert persisted_rates["RUB"] == Decimal("91")
    assert result.rate == Decimal("91")
    assert result.amount == Decimal("910")
    assert result.stale is False
    assert result.source == "open.er-api.com"


async def test_today_miss_upstream_failure_falls_back_to_last_known(patch_session, stub_helpers):
    today = date.today()
    yesterday = date.fromordinal(today.toordinal() - 1)
    stub_helpers.upstream = None
    stub_helpers.last[("USD", "RUB")] = _RateHit(
        rate=Decimal("89"), date_used=yesterday, source="seed"
    )
    result = await currency_convert(Decimal("100"), "USD", "RUB")
    assert stub_helpers.fetch_calls == 1
    assert result.amount == Decimal("8900")
    assert result.stale is True
    assert result.date_used == yesterday


async def test_historical_miss_does_not_fetch_upstream(patch_session, stub_helpers):
    historical = date(2025, 1, 15)
    stub_helpers.last[("USD", "RUB")] = _RateHit(
        rate=Decimal("88"), date_used=date(2025, 1, 10), source="seed"
    )
    result = await currency_convert(Decimal("100"), "USD", "RUB", on_date=historical)
    assert stub_helpers.fetch_calls == 0
    assert result.stale is True
    assert result.date_used == date(2025, 1, 10)
    assert result.amount == Decimal("8800")


async def test_historical_no_data_raises(patch_session, stub_helpers):
    historical = date(2020, 1, 1)
    with pytest.raises(CurrencyConvertError, match="no rate available"):
        await currency_convert(Decimal("1"), "USD", "RUB", on_date=historical)
    assert stub_helpers.fetch_calls == 0


async def test_inverted_last_known_when_only_reverse_pair_exists(patch_session, stub_helpers):
    historical = date(2025, 6, 1)
    stub_helpers.last[("RUB", "USD")] = _RateHit(
        rate=Decimal("0.0125"), date_used=date(2025, 5, 30), source="seed"
    )
    result = await currency_convert(Decimal("100"), "USD", "RUB", on_date=historical)
    assert result.stale is True
    assert result.rate == Decimal("80")
    assert result.amount == Decimal("8000")


# ---------------------------- upstream fetch ----------------------------


def _mock_httpx(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    class _Client(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ccm.httpx, "AsyncClient", _Client)


async def test_fetch_upstream_parses_success(monkeypatch):
    body = {
        "result": "success",
        "base_code": "USD",
        "time_last_update_unix": 1745452800,  # 2025-04-24 UTC
        "rates": {"USD": 1, "RUB": 92.5, "EUR": 0.93, "lol": 1, "BAD": "oops"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/USD")
        return httpx.Response(200, json=body)

    _mock_httpx(monkeypatch, handler)
    fetched = await ccm._fetch_upstream("USD")
    assert fetched is not None
    fetched_date, rates, source = fetched
    assert fetched_date == date(2025, 4, 24)
    assert rates["RUB"] == Decimal("92.5")
    assert rates["EUR"] == Decimal("0.93")
    assert "lol" not in rates  # not ISO 4217 (lowercase)
    assert "BAD" not in rates  # non-numeric value
    assert source == "open.er-api.com"


async def test_fetch_upstream_http_error_returns_none(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    _mock_httpx(monkeypatch, handler)
    assert await ccm._fetch_upstream("USD") is None


async def test_fetch_upstream_provider_failure_returns_none(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": "error", "rates": {}})

    _mock_httpx(monkeypatch, handler)
    assert await ccm._fetch_upstream("XYZ") is None


async def test_fetch_upstream_network_error_returns_none(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _mock_httpx(monkeypatch, handler)
    assert await ccm._fetch_upstream("USD") is None


async def test_fetch_upstream_skips_zero_or_negative_rates(monkeypatch):
    body = {
        "result": "success",
        "base_code": "USD",
        "time_last_update_unix": 1745452800,
        "rates": {"RUB": 0, "EUR": -1, "GBP": 0.79},
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    _mock_httpx(monkeypatch, handler)
    fetched = await ccm._fetch_upstream("USD")
    assert fetched is not None
    _, rates, _ = fetched
    assert rates == {"GBP": Decimal("0.79")}
