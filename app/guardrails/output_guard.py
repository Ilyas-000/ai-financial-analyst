"""Output guardrail (I-11).

Runs in ``ChatService`` after the graph finished, on the assembled answer
plus its sources and (optional) ``suggested_action``. Three checks:

1. **Cross-tenant leak** — the answer mentions another tenant's display name
   or INN. Per design.md §8 and plan.md §1.6 this is the hard mandatory check.
   Reaction (per R2, 2026-05-27): replace ``answer`` with a neutral Russian
   notice, drop ``sources`` and ``suggested_action``, append an ``error``
   marker, and write an ``audit_log`` row with ``severity="high"`` so the
   incident is durably recorded.
2. **PII masking** — card numbers in the answer become ``**** <last4>``.
   Non-blocking; we just rewrite the text. Same Luhn validator as input guard.
3. **Suggested-action invariant** — any ``suggested_action`` that lacks
   ``requires_confirmation=True`` is dropped (defensive: builders enforce
   this, but a hand-rolled future path could regress).

Foreign-tenant detection is conservative: name match is whole-word
case-insensitive, INN match is exact 10/12-digit literal. Bare ``company_id``
integers are deliberately NOT scanned — false-positive rate on financial
text is too high.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.db.models import AuditLog
from app.db.session import get_sessionmaker
from app.guardrails.patterns import iter_card_spans, mask_card_match
from app.guardrails.tenant_index import TenantEntry, load_tenant_index

logger = logging.getLogger(__name__)

_REJECTION_ANSWER = (
    "Запрос отклонён системой безопасности: ответ содержал данные другой "
    "компании, доступ к которым у вас отсутствует. Если это был легитимный "
    "запрос, обратитесь к администратору."
)


@dataclass(frozen=True)
class OutputGuardResult:
    answer: str
    sources: list[dict[str, Any]]
    suggested_action: dict[str, Any] | None
    rejected: bool = False
    rejection_reason: str | None = None
    masked_cards: int = 0
    dropped_action: bool = False
    foreign_hits: list[str] = field(default_factory=list)


def _foreign_tenants(index: list[TenantEntry], user_company_id: int) -> list[TenantEntry]:
    return [t for t in index if t.company_id != user_company_id]


def _scan_foreign_hits(answer: str, foreign: list[TenantEntry]) -> list[str]:
    """Return a list of `kind:value` markers for every foreign-tenant hit."""
    hits: list[str] = []
    for tenant in foreign:
        name = tenant.name.strip()
        if name and re.search(rf"\b{re.escape(name)}\b", answer, flags=re.IGNORECASE):
            hits.append(f"name:{name}")
        inn = tenant.inn.strip()
        if inn and re.search(rf"(?<!\d){re.escape(inn)}(?!\d)", answer):
            hits.append(f"inn:{inn}")
    return hits


def _mask_cards_in_text(text: str) -> tuple[str, int]:
    spans = iter_card_spans(text)
    if not spans:
        return text, 0
    buf = text
    for start, end, digits in reversed(spans):
        buf = buf[:start] + mask_card_match(digits) + buf[end:]
    return buf, len(spans)


async def _write_audit(
    *,
    company_id: int,
    user_role: str,
    action: str,
    severity: str,
    details: dict[str, Any],
) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session, session.begin():
        session.add(
            AuditLog(
                company_id=company_id,
                user_role=user_role,
                action=action,
                sql_text=None,
                rows_returned=None,
                severity=severity,
                details=details,
                created_at=datetime.now(UTC),
            )
        )


async def apply_output_guard(
    *,
    answer: str,
    sources: list[dict[str, Any]],
    suggested_action: dict[str, Any] | None,
    user_company_id: int,
    user_role: str,
    thread_id: str,
) -> OutputGuardResult:
    """Validate the graph's final output and rewrite it where needed.

    Cross-tenant hits short-circuit: the answer is replaced, sources and
    action are dropped, and an audit row is written. Otherwise we mask cards
    in the answer and enforce the ``requires_confirmation`` invariant on the
    suggested action.
    """
    index = await load_tenant_index()
    foreign = _foreign_tenants(index, user_company_id)
    hits = _scan_foreign_hits(answer, foreign)

    if hits:
        await _write_audit(
            company_id=user_company_id,
            user_role=user_role,
            action="output_blocked_cross_tenant",
            severity="high",
            details={
                "thread_id": thread_id,
                "foreign_hits": hits,
                "answer_excerpt": answer[:500],
            },
        )
        logger.warning("output guard blocked cross-tenant leak: thread=%s hits=%s", thread_id, hits)
        return OutputGuardResult(
            answer=_REJECTION_ANSWER,
            sources=[],
            suggested_action=None,
            rejected=True,
            rejection_reason="cross_tenant_leak",
            foreign_hits=hits,
        )

    masked_answer, n_cards = _mask_cards_in_text(answer)

    cleaned_action = suggested_action
    dropped_action = False
    if suggested_action is not None and not bool(suggested_action.get("requires_confirmation")):
        cleaned_action = None
        dropped_action = True
        logger.warning(
            "output guard dropped suggested_action without confirmation: thread=%s kind=%s",
            thread_id,
            suggested_action.get("kind"),
        )

    return OutputGuardResult(
        answer=masked_answer,
        sources=list(sources),
        suggested_action=cleaned_action,
        rejected=False,
        rejection_reason=None,
        masked_cards=n_cards,
        dropped_action=dropped_action,
    )
