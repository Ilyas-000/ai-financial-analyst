"""FX conversion tool with cache-first lookup and a single fallback path.

Resolution order for ``currency_convert(amount, from, to, on_date=None)``:

1. ``from == to`` — identity, no DB / network.
2. Exact pair (``base, quote, target_date``) in ``currency_rates``.
3. Inverted pair (``to, from, target_date``) — used as ``1 / rate``.
4. **Only if ``on_date is None``** — fetch ``open.er-api.com`` for ``from`` as
   base, persist all returned rates for that day, re-query the exact pair.
5. Last-known fallback: max date ``<= target_date`` for the same pair (or its
   inverse). Result carries ``stale=True``.
6. Otherwise — :class:`CurrencyConvertError`.

The free upstream API only returns *latest* rates, so historical-date misses
never go to the network; they fall straight to the last-known fallback.
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import CurrencyRate
from app.db.session import get_sessionmaker

_CCY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")
_HTTP_TIMEOUT_S: Final[float] = 10.0


class CurrencyConvertError(ValueError):
    """Raised when conversion cannot be resolved (bad code, no rate, upstream)."""


@dataclass(frozen=True)
class ConversionResult:
    amount: Decimal
    rate: Decimal
    date_used: date
    stale: bool
    source: str
    from_ccy: str
    to_ccy: str


@dataclass(frozen=True)
class _RateHit:
    rate: Decimal
    date_used: date
    source: str


async def currency_convert(
    amount: Decimal | int | float | str,
    from_ccy: str,
    to_ccy: str,
    *,
    on_date: date | None = None,
) -> ConversionResult:
    """Convert ``amount`` from ``from_ccy`` to ``to_ccy`` on ``on_date`` (or today)."""
    from_ccy = _normalize_code(from_ccy, label="from_ccy")
    to_ccy = _normalize_code(to_ccy, label="to_ccy")
    decimal_amount = _coerce_amount(amount)
    target_date = on_date if on_date is not None else date.today()
    is_today_request = on_date is None

    if from_ccy == to_ccy:
        return ConversionResult(
            amount=decimal_amount,
            rate=Decimal("1"),
            date_used=target_date,
            stale=False,
            source="identity",
            from_ccy=from_ccy,
            to_ccy=to_ccy,
        )

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session, session.begin():
        hit, stale = await _resolve_rate(
            session,
            from_ccy=from_ccy,
            to_ccy=to_ccy,
            target_date=target_date,
            is_today_request=is_today_request,
        )

    if hit is None:
        raise CurrencyConvertError(
            f"no rate available for {from_ccy}->{to_ccy} on {target_date.isoformat()}"
        )

    return ConversionResult(
        amount=decimal_amount * hit.rate,
        rate=hit.rate,
        date_used=hit.date_used,
        stale=stale,
        source=hit.source,
        from_ccy=from_ccy,
        to_ccy=to_ccy,
    )


async def _resolve_rate(
    session: AsyncSession,
    *,
    from_ccy: str,
    to_ccy: str,
    target_date: date,
    is_today_request: bool,
) -> tuple[_RateHit | None, bool]:
    # Step 1: exact pair on target date.
    hit = await _query_exact(session, from_ccy, to_ccy, target_date)
    if hit is not None:
        return hit, False

    # Step 2: inverted pair on target date.
    inv = await _query_exact(session, to_ccy, from_ccy, target_date)
    if inv is not None:
        return _invert(inv), False

    # Step 3: upstream — only if "today", upstream has no historical rates.
    if is_today_request:
        fetched = await _fetch_upstream(from_ccy)
        if fetched is not None:
            fetched_date, rates, source = fetched
            await _persist_rates(session, from_ccy, fetched_date, rates, source)
            hit = await _query_exact(session, from_ccy, to_ccy, fetched_date)
            if hit is not None:
                return hit, False

    # Step 4: last-known fallback (direct or inverted).
    last = await _query_last_known(session, from_ccy, to_ccy, target_date)
    if last is not None:
        return last, True
    last_inv = await _query_last_known(session, to_ccy, from_ccy, target_date)
    if last_inv is not None:
        return _invert(last_inv), True

    return None, False


async def _query_exact(
    session: AsyncSession, base: str, quote: str, target_date: date
) -> _RateHit | None:
    stmt = select(CurrencyRate).where(
        CurrencyRate.base == base,
        CurrencyRate.quote == quote,
        CurrencyRate.date == target_date,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return _RateHit(rate=row.rate, date_used=row.date, source=row.source)


async def _query_last_known(
    session: AsyncSession, base: str, quote: str, target_date: date
) -> _RateHit | None:
    stmt = (
        select(CurrencyRate)
        .where(
            CurrencyRate.base == base,
            CurrencyRate.quote == quote,
            CurrencyRate.date <= target_date,
        )
        .order_by(CurrencyRate.date.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return _RateHit(rate=row.rate, date_used=row.date, source=row.source)


async def _persist_rates(
    session: AsyncSession,
    base: str,
    fetched_date: date,
    rates: dict[str, Decimal],
    source: str,
) -> None:
    if not rates:
        return
    payload: list[dict[str, Any]] = [
        {
            "base": base,
            "quote": quote,
            "date": fetched_date,
            "rate": rate,
            "source": source,
        }
        for quote, rate in rates.items()
    ]
    stmt = pg_insert(CurrencyRate).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=[CurrencyRate.base, CurrencyRate.quote, CurrencyRate.date],
        set_={"rate": stmt.excluded.rate, "source": stmt.excluded.source},
    )
    await session.execute(stmt)


async def _fetch_upstream(base: str) -> tuple[date, dict[str, Decimal], str] | None:
    """Hit ``open.er-api.com``. Returns ``None`` on any upstream failure."""
    settings = get_settings()
    url = f"{settings.fx_base_url.rstrip('/')}/{base}"
    source = urlsplit(settings.fx_base_url).hostname or "open.er-api.com"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            response = await client.get(url)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if body.get("result") != "success":
        return None

    raw_rates = body.get("rates")
    if not isinstance(raw_rates, dict):
        return None
    rates: dict[str, Decimal] = {}
    for quote, value in raw_rates.items():
        if not isinstance(quote, str) or not _CCY_RE.match(quote):
            continue
        try:
            decimal_rate = Decimal(str(value))
        except (InvalidOperation, TypeError):
            continue
        if decimal_rate <= 0:
            continue
        rates[quote] = decimal_rate

    fetched_date = _parse_upstream_date(body)
    return fetched_date, rates, source


def _parse_upstream_date(body: dict[str, Any]) -> date:
    unix_value = body.get("time_last_update_unix")
    if isinstance(unix_value, int | float) and unix_value > 0:
        return datetime.fromtimestamp(unix_value, tz=UTC).date()
    return date.today()


def _invert(hit: _RateHit) -> _RateHit:
    if hit.rate == 0:
        # Stored rate of 0 makes no economic sense; treat as no data.
        raise CurrencyConvertError("invalid stored rate: 0")
    inverted = Decimal("1") / hit.rate
    return _RateHit(rate=inverted, date_used=hit.date_used, source=hit.source)


def _normalize_code(code: str, *, label: str) -> str:
    if not isinstance(code, str):
        raise CurrencyConvertError(f"{label} must be a string")
    upper = code.strip().upper()
    if not _CCY_RE.match(upper):
        raise CurrencyConvertError(f"{label}={code!r} is not a valid ISO 4217 code")
    return upper


def _coerce_amount(amount: Decimal | int | float | str) -> Decimal:
    if isinstance(amount, Decimal):
        return amount
    if isinstance(amount, bool):
        raise CurrencyConvertError("amount must be a number, not bool")
    if isinstance(amount, int):
        return Decimal(amount)
    if isinstance(amount, float):
        return Decimal(repr(amount))
    if isinstance(amount, str):
        try:
            return Decimal(amount)
        except InvalidOperation as exc:
            raise CurrencyConvertError(f"amount={amount!r} is not numeric") from exc
    raise CurrencyConvertError(f"amount has unsupported type: {type(amount).__name__}")
