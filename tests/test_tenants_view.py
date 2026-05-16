"""Unit tests for ``app.services.tenants_view``.

We mock the SQLAlchemy session so the test doesn't need a live database — the
view's only logic is shaping rows into ``TenantOption`` and ordering them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import tenants_view
from app.services.tenants_view import TenantOption, list_tenants_for_ui


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.executed.append(statement)
        return _FakeResult(self._rows)


class _FakeSessionmaker:
    def __init__(self, rows):
        self._rows = rows

    def __call__(self):
        return _FakeSession(self._rows)


async def test_list_tenants_for_ui_returns_options_in_id_order():
    rows = [
        SimpleNamespace(id=1, name="ACME LLC"),
        SimpleNamespace(id=2, name="Ostrovok-mock"),
        SimpleNamespace(id=3, name="CheckScan-mock"),
    ]

    with patch.object(tenants_view, "get_sessionmaker", return_value=_FakeSessionmaker(rows)):
        result = await list_tenants_for_ui()

    assert result == [
        TenantOption(company_id=1, name="ACME LLC", slug="acme"),
        TenantOption(company_id=2, name="Ostrovok-mock", slug="ostrovok"),
        TenantOption(company_id=3, name="CheckScan-mock", slug="checkscan"),
    ]


async def test_list_tenants_for_ui_returns_empty_list_when_no_companies():
    with patch.object(tenants_view, "get_sessionmaker", return_value=_FakeSessionmaker([])):
        result = await list_tenants_for_ui()
    assert result == []


async def test_list_tenants_for_ui_propagates_db_errors():
    """Errors from the session aren't swallowed — the UI surfaces a message."""

    class _ExplodingSessionmaker:
        def __call__(self):
            class _Sess:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                execute = AsyncMock(side_effect=RuntimeError("db down"))

            return _Sess()

    with patch.object(tenants_view, "get_sessionmaker", return_value=_ExplodingSessionmaker()):
        raised = False
        try:
            await list_tenants_for_ui()
        except RuntimeError as exc:
            raised = True
            assert "db down" in str(exc)
        assert raised, "expected RuntimeError to propagate"
