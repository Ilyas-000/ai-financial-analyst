"""Shared regex patterns + Luhn validator for input/output guards (I-11).

Pure data and stateless helpers — no I/O. Both ``input_guard`` (masks PII
before the question reaches the LLM) and ``output_guard`` (masks card numbers
in the final answer) consume from here so the masking is consistent.

Patterns are intentionally conservative: false-positive masking is fine —
the user's question / final answer remains readable — but a false negative
leaks PII or a card number into the LLM trace or back to the UI.
"""

import re

# --- Prompt-injection blocklist ------------------------------------------------
# Substring matches, case-insensitive. We keep this short and obvious; a real
# classifier would live behind the same interface. RU + EN coverage because
# user questions arrive in Russian and most jailbreak templates leak from EN.
INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard the above",
    "system:",
    "system prompt",
    "you are now",
    "act as",
    "developer mode",
    "jailbreak",
    "забудь предыдущие",
    "забудь все предыдущие",
    "игнорируй инструкции",
    "игнорируй предыдущие",
    "ты теперь",
    "представь, что ты",
    "представь что ты",
    "режим разработчика",
)


def find_injection_hit(text: str) -> str | None:
    """Return the first blocklist phrase found in ``text`` (case-insensitive), or None."""
    lowered = text.lower()
    for phrase in INJECTION_PATTERNS:
        if phrase in lowered:
            return phrase
    return None


# --- Card numbers --------------------------------------------------------------
# Match runs of 13–19 digits possibly separated by spaces or hyphens. Luhn
# checksum is applied after extraction to drop false positives (e.g. a
# transactions.id like ``20250416000123`` would syntactically match).
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum. ``digits`` must contain only ASCII digits."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = ord(char) - 48
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iter_card_spans(text: str) -> list[tuple[int, int, str]]:
    """Find Luhn-valid card-number spans in ``text``.

    Returns ``[(start, end, digits)]`` sorted left-to-right. Only spans whose
    extracted digits pass Luhn are reported — keeps order/payment ids out.
    """
    spans: list[tuple[int, int, str]] = []
    for match in _CARD_RE.finditer(text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            spans.append((match.start(), match.end(), digits))
    return spans


def mask_card_match(digits: str) -> str:
    """Mask a card number to ``**** <last4>``."""
    return f"**** {digits[-4:]}"


# --- Russian INN ---------------------------------------------------------------
# Legal entity = 10 digits, individual = 12 digits. Checksum validation removes
# the bulk of false positives (10-digit transaction ids etc.). Output guard
# also compares masked / extracted INNs against the tenant catalogue, so
# checksum here is a precision boost, not a security boundary.
_INN_RE = re.compile(r"(?<!\d)\d{10}(?!\d)|(?<!\d)\d{12}(?!\d)")
_INN10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _checksum(digits: str, weights: tuple[int, ...]) -> int:
    total = sum(int(digits[i]) * weights[i] for i in range(len(weights)))
    return (total % 11) % 10


def inn_valid(digits: str) -> bool:
    if len(digits) == 10:
        return _checksum(digits, _INN10_WEIGHTS) == int(digits[9])
    if len(digits) == 12:
        c11 = _checksum(digits, _INN12_WEIGHTS_1)
        c12 = _checksum(digits, _INN12_WEIGHTS_2)
        return c11 == int(digits[10]) and c12 == int(digits[11])
    return False


def iter_inn_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _INN_RE.finditer(text):
        digits = match.group(0)
        if inn_valid(digits):
            spans.append((match.start(), match.end(), digits))
    return spans


# --- Russian passport ----------------------------------------------------------
# ``ХХХХ ХХХХХХ`` — series (4 digits) + number (6 digits). Optional separator
# (space, hyphen, or run of whitespace). No checksum in the standard.
_PASSPORT_RE = re.compile(r"(?<!\d)\d{4}[ \- ]?\d{6}(?!\d)")


def iter_passport_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in _PASSPORT_RE.finditer(text)]
