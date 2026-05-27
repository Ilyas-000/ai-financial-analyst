"""Input guardrail (I-11).

Runs in ``ChatService`` before the user's question reaches the LangGraph
supervisor. Three responsibilities, in order:

1. Reject over-length input (cheap DoS / context-pollution guard).
2. Detect prompt-injection patterns and block.
3. Mask PII (card numbers via Luhn, Russian INN, passport) so neither the LLM
   nor the trace store (Langfuse) sees raw secrets.

Hard rejections raise :class:`InputGuardError`; ChatService converts that into
a user-facing ``ChatResult`` with ``route="blocked"`` and an explanatory
Russian message. Masking is non-fatal and reported via ``InputGuardResult``.
"""

from dataclasses import dataclass, field
from typing import Literal

from app.guardrails.patterns import (
    find_injection_hit,
    iter_card_spans,
    iter_inn_spans,
    iter_passport_spans,
    mask_card_match,
)

DEFAULT_MAX_CHARS = 2000

BlockReason = Literal["too_long", "prompt_injection"]


class InputGuardError(Exception):
    """Raised when the question must not reach the LLM."""

    def __init__(self, message: str, *, reason: BlockReason, detail: str = "") -> None:
        super().__init__(message)
        self.reason: BlockReason = reason
        self.detail = detail


@dataclass(frozen=True)
class InputGuardResult:
    question: str
    masked_cards: int = 0
    masked_inns: int = 0
    masked_passports: int = 0
    flags: list[str] = field(default_factory=list)


def _mask_spans(text: str, spans: list[tuple[int, int, str]], replacer) -> tuple[str, int]:
    """Replace ``spans`` in ``text`` right-to-left so earlier offsets stay valid."""
    if not spans:
        return text, 0
    buf = text
    for start, end, raw in reversed(spans):
        buf = buf[:start] + replacer(raw) + buf[end:]
    return buf, len(spans)


def apply_input_guard(
    question: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> InputGuardResult:
    """Validate and sanitise a user question before it enters the graph.

    Raises :class:`InputGuardError` for hard blocks (too long, injection hit).
    Otherwise returns the masked question plus mask-count bookkeeping for the
    observability / audit trail.
    """
    if not isinstance(question, str):
        raise InputGuardError(
            "Вопрос должен быть текстом.", reason="too_long", detail="non-string input"
        )

    if len(question) > max_chars:
        raise InputGuardError(
            f"Вопрос слишком длинный (максимум {max_chars} символов).",
            reason="too_long",
            detail=f"length={len(question)}",
        )

    hit = find_injection_hit(question)
    if hit is not None:
        raise InputGuardError(
            "Запрос отклонён системой безопасности: обнаружена попытка "
            "переопределить инструкции ассистента.",
            reason="prompt_injection",
            detail=f"matched={hit!r}",
        )

    flags: list[str] = []
    masked, n_cards = _mask_spans(question, iter_card_spans(question), mask_card_match)
    if n_cards:
        flags.append(f"masked_cards={n_cards}")

    masked, n_inns = _mask_spans(masked, iter_inn_spans(masked), lambda _digits: "<masked-inn>")
    if n_inns:
        flags.append(f"masked_inns={n_inns}")

    masked, n_passports = _mask_spans(
        masked, iter_passport_spans(masked), lambda _raw: "<masked-passport>"
    )
    if n_passports:
        flags.append(f"masked_passports={n_passports}")

    return InputGuardResult(
        question=masked,
        masked_cards=n_cards,
        masked_inns=n_inns,
        masked_passports=n_passports,
        flags=flags,
    )
