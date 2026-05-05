"""Draft action builder.

The agent never executes a side-effectful action itself. After the answer is
ready, Finalize may attach a *draft* — a structured suggestion that the UI
renders with an explicit "confirm to run" affordance.

Four kinds, matching design.md §7:
    export_report, open_ticket, prepare_act, highlight_discrepancy

Every draft carries ``requires_confirmation=True`` as an invariant. The dict
shape returned by :meth:`DraftAction.as_payload` matches the
``suggested_action`` field of ``AgentState`` (design.md §4) — a flat
``{kind, title, payload, requires_confirmation}`` ready to drop into state.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Literal

DraftKind = Literal["export_report", "open_ticket", "prepare_act", "highlight_discrepancy"]
TicketSeverity = Literal["low", "medium", "high"]
ReportFormat = Literal["csv", "xlsx", "pdf"]

_CURRENCY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")
_VALID_FORMATS: Final[frozenset[str]] = frozenset({"csv", "xlsx", "pdf"})
_VALID_SEVERITIES: Final[frozenset[str]] = frozenset({"low", "medium", "high"})


class DraftActionError(ValueError):
    """Raised when a draft action cannot be assembled (bad input, unknown kind)."""


@dataclass(frozen=True)
class DraftAction:
    kind: DraftKind
    title: str
    payload: dict[str, Any]
    requires_confirmation: bool = True

    def as_payload(self) -> dict[str, Any]:
        """Return a flat dict suitable for ``AgentState.suggested_action``."""
        return {
            "kind": self.kind,
            "title": self.title,
            "payload": dict(self.payload),
            "requires_confirmation": self.requires_confirmation,
        }


def build_export_report(
    *,
    report_type: str,
    period_start: date,
    period_end: date,
    currency: str,
    fmt: str,
) -> DraftAction:
    """Draft for ``export_report``. Used when the user asks for a report file."""
    _require_nonempty_str("report_type", report_type)
    _require_period(period_start, period_end)
    currency_norm = _normalize_currency(currency)
    if fmt not in _VALID_FORMATS:
        raise DraftActionError(f"unsupported report format: {fmt!r}")
    payload = {
        "report_type": report_type,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "currency": currency_norm,
        "format": fmt,
    }
    title = (
        f"Сформировать отчёт «{report_type}» за "
        f"{period_start.isoformat()} — {period_end.isoformat()}"
    )
    return DraftAction(kind="export_report", title=title, payload=payload)


def build_open_ticket(
    *,
    topic: str,
    severity: str,
    summary: str,
    related_ids: dict[str, list[int]] | None = None,
) -> DraftAction:
    """Draft for ``open_ticket``. Used for incidents and escalations."""
    _require_nonempty_str("topic", topic)
    _require_nonempty_str("summary", summary)
    if severity not in _VALID_SEVERITIES:
        raise DraftActionError(f"unsupported severity: {severity!r}")
    payload = {
        "topic": topic,
        "severity": severity,
        "summary": summary,
        "related_ids": _normalize_related_ids(related_ids),
    }
    return DraftAction(
        kind="open_ticket",
        title=f"Открыть тикет: {topic}",
        payload=payload,
    )


def build_prepare_act(
    *,
    counterparty_name: str,
    period_start: date,
    period_end: date,
    payout_ids: list[int],
    counterparty_inn: str | None = None,
) -> DraftAction:
    """Draft for ``prepare_act``. Used for closing documents with counterparties."""
    _require_nonempty_str("counterparty_name", counterparty_name)
    _require_period(period_start, period_end)
    if not payout_ids:
        raise DraftActionError("payout_ids must not be empty")
    payload = {
        "counterparty_name": counterparty_name,
        "counterparty_inn": counterparty_inn,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "payout_ids": _normalize_int_list("payout_ids", payout_ids),
    }
    return DraftAction(
        kind="prepare_act",
        title=f"Подготовить акт для {counterparty_name}",
        payload=payload,
    )


def build_highlight_discrepancy(
    *,
    dimension: str,
    expected: Decimal | int | float | str,
    actual: Decimal | int | float | str,
    refs: dict[str, list[int]] | None = None,
) -> DraftAction:
    """Draft for ``highlight_discrepancy``. Used to flag mismatched figures."""
    _require_nonempty_str("dimension", dimension)
    payload = {
        "dimension": dimension,
        "expected": _format_amount(expected),
        "actual": _format_amount(actual),
        "refs": _normalize_related_ids(refs),
    }
    return DraftAction(
        kind="highlight_discrepancy",
        title=f"Расхождение по «{dimension}»",
        payload=payload,
    )


_BUILDERS: Final[dict[str, Any]] = {
    "export_report": build_export_report,
    "open_ticket": build_open_ticket,
    "prepare_act": build_prepare_act,
    "highlight_discrepancy": build_highlight_discrepancy,
}


def build_draft_action(kind: str, /, **context: Any) -> DraftAction:
    """Dispatch to the per-kind builder. Unknown kinds raise."""
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise DraftActionError(f"unknown draft action kind: {kind!r}")
    return builder(**context)


# ---------------------------- helpers ----------------------------


def _require_nonempty_str(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DraftActionError(f"{name} must be a non-empty string")


def _require_period(start: Any, end: Any) -> None:
    if not isinstance(start, date) or not isinstance(end, date):
        raise DraftActionError("period_start and period_end must be date instances")
    if start > end:
        raise DraftActionError(
            f"period_start={start.isoformat()} is after period_end={end.isoformat()}"
        )


def _normalize_currency(currency: Any) -> str:
    if not isinstance(currency, str):
        raise DraftActionError("currency must be a string")
    upper = currency.strip().upper()
    if not _CURRENCY_RE.match(upper):
        raise DraftActionError(f"currency={currency!r} is not a valid ISO 4217 code")
    return upper


def _normalize_int_list(name: str, ids: Any) -> list[int]:
    if not isinstance(ids, list):
        raise DraftActionError(f"{name} must be a list of int")
    cleaned: list[int] = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DraftActionError(f"{name} must contain int values")
        cleaned.append(value)
    return cleaned


def _normalize_related_ids(refs: dict[str, list[int]] | None) -> dict[str, list[int]]:
    if refs is None:
        return {}
    if not isinstance(refs, dict):
        raise DraftActionError("related_ids/refs must be a dict[str, list[int]]")
    result: dict[str, list[int]] = {}
    for key, values in refs.items():
        if not isinstance(key, str) or not key:
            raise DraftActionError("related_ids/refs keys must be non-empty strings")
        result[key] = _normalize_int_list(f"related_ids[{key}]", values)
    return result


def _format_amount(value: Decimal | int | float | str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise DraftActionError("amount string must not be empty")
        return value
    if isinstance(value, bool):
        raise DraftActionError("amount must be a number, not bool")
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, int):
        return str(Decimal(value))
    if isinstance(value, float):
        try:
            return str(Decimal(repr(value)))
        except InvalidOperation as exc:
            raise DraftActionError(f"amount={value!r} is not a finite number") from exc
    raise DraftActionError(f"unsupported amount type: {type(value).__name__}")
