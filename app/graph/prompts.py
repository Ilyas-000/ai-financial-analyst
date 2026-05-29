"""Two-section prompt loader shared across graph nodes.

Each prompt file uses ``=== SYSTEM ===`` / ``=== USER ===`` markers; the
loader returns the two halves stripped, ready for ``str.format`` with
node-specific placeholders.
"""

from functools import lru_cache
from pathlib import Path

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER ==="

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=32)
def load_two_section_prompt(name: str) -> tuple[str, str]:
    path = PROMPTS_DIR / name
    raw = path.read_text(encoding="utf-8")
    if _SYSTEM_MARKER not in raw or _USER_MARKER not in raw:
        raise ValueError(f"{path} must contain both {_SYSTEM_MARKER!r} and {_USER_MARKER!r}")
    _, _, after_system = raw.partition(_SYSTEM_MARKER)
    system_part, _, user_part = after_system.partition(_USER_MARKER)
    return system_part.strip(), user_part.strip()


def strip_code_fences(text: str) -> str:
    """Remove a single leading/trailing markdown code fence, if present.

    Local 3B/7B models routinely wrap their output in ```...``` even when
    asked for raw SQL/JSON/text. Callers that need extra trimming (a trailing
    ``;`` for SQL, surrounding quotes for the condenser) layer that on top of
    this base helper.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()
