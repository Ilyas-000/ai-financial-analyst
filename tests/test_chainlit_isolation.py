"""Static invariant: the Chainlit UI layer must not reach into the AI core.

``rules.md`` §2.1: ``app/ui/chainlit_app.py`` may import only from
``app.services``, ``app.config``, ``app.main`` (for the FastAPI app handle),
plus third-party packages and the standard library. Importing from
``app.graph``, ``app.rag``, ``app.tools`` or ``app.db`` would bypass the
single ``ChatService`` entrypoint and couple the UI to the agent internals —
deleting the UI would no longer be free.

We parse the file (instead of grepping) so block comments and string
literals can mention forbidden names without tripping the check.

Plus two cheap sanity tests for the in-module profile resolver — wrong
profile names should not crash the handlers.
"""

import ast
from pathlib import Path

from app.ui.chainlit_app import _PROFILES, ChatProfileSpec, _resolve_profile

UI_FILE = Path(__file__).resolve().parents[1] / "app" / "ui" / "chainlit_app.py"

_FORBIDDEN_ROOTS = ("app.graph", "app.rag", "app.tools", "app.db")
_ALLOWED_INTERNAL = ("app.services", "app.config", "app.main")


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_chainlit_app_imports_only_through_services():
    source = UI_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _imported_modules(tree)

    forbidden_hits = [m for m in imported if m.startswith(_FORBIDDEN_ROOTS)]
    assert not forbidden_hits, (
        f"chainlit_app.py must not import {_FORBIDDEN_ROOTS}; "
        f"found: {sorted(set(forbidden_hits))}. "
        "Route the access through app.services instead."
    )

    # Sanity: every ``app.*`` import lands in the allowlist; new sibling
    # packages added under app.* should opt in explicitly.
    app_imports = [m for m in imported if m.startswith("app.")]
    for module in app_imports:
        assert any(module.startswith(prefix) for prefix in _ALLOWED_INTERNAL), (
            f"chainlit_app.py imports {module!r} which is outside the allowlist "
            f"{_ALLOWED_INTERNAL}. Either route through app.services or amend the rule."
        )


def test_profiles_have_unique_names_and_valid_roles():
    names = [profile.name for profile in _PROFILES]
    assert len(names) == len(set(names)), "chat profile names must be unique"
    valid_roles = {"cfo", "finance_manager", "accountant", "auditor"}
    for profile in _PROFILES:
        assert isinstance(profile, ChatProfileSpec)
        assert profile.role in valid_roles, f"profile {profile.name!r} has unknown role"
        assert profile.tenant_slug, f"profile {profile.name!r} missing tenant slug"


def test_resolve_profile_round_trip_and_misses():
    first = _PROFILES[0]
    assert _resolve_profile(first.name) is first
    assert _resolve_profile("does not exist") is None
    assert _resolve_profile(None) is None
    assert _resolve_profile("") is None
