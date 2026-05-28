"""I-12 audit-log enrichment: every audit row must carry ``user_id``.

We don't spin up Postgres for unit tests — we monkey-patch
``get_sessionmaker`` so ``session.add(AuditLog(...))`` lands in an in-process
list and the assertions can inspect the ORM instance directly. This pins
down the contract between ``execute_guarded`` and the AuditLog model
without requiring the real DB.
"""

from contextlib import asynccontextmanager
from typing import Any

import pytest

from app.db.models import AuditLog
from app.tools import sql_executor
from app.tools.sql_executor import execute_guarded


@pytest.fixture
def captured_audit(monkeypatch: pytest.MonkeyPatch) -> list[AuditLog]:
    """Capture every AuditLog row execute_guarded tries to persist."""
    rows: list[AuditLog] = []

    class _FakeSession:
        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        @asynccontextmanager
        async def begin(self):
            yield

        def add(self, instance: Any) -> None:
            if isinstance(instance, AuditLog):
                rows.append(instance)

        async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            class _Result:
                def mappings(self) -> Any:
                    return self

                def all(self) -> list[dict[str, Any]]:
                    return []

            return _Result()

    def _fake_sessionmaker():
        def factory() -> _FakeSession:
            return _FakeSession()

        return factory

    monkeypatch.setattr(sql_executor, "get_sessionmaker", _fake_sessionmaker)
    return rows


@pytest.mark.asyncio
async def test_audit_row_carries_user_id_on_success(captured_audit: list[AuditLog]) -> None:
    """A successful SELECT writes one audit row with user_id populated."""
    result = await execute_guarded(
        "SELECT id FROM transactions WHERE company_id = 1 LIMIT 10",
        user_id="tenant:1:finance_manager",
        user_role="finance_manager",
        company_id=1,
        thread_id="t-abc",
    )

    assert result.rows_returned == 0  # FakeSession returns no rows
    assert len(captured_audit) == 1
    row = captured_audit[0]
    assert row.user_id == "tenant:1:finance_manager"
    assert row.user_role == "finance_manager"
    assert row.company_id == 1
    assert row.action == "sql_executed"
    assert row.details is not None
    assert row.details["thread_id"] == "t-abc"
    assert "duration_ms" in row.details


@pytest.mark.asyncio
async def test_audit_row_carries_user_id_on_guard_rejection(
    captured_audit: list[AuditLog],
) -> None:
    """Guard rejection still produces an audit row with user_id."""
    from app.tools.sql_guard import GuardRejection

    with pytest.raises(GuardRejection):
        await execute_guarded(
            "UPDATE transactions SET amount = 0",
            user_id="tenant:2:cfo",
            user_role="cfo",
            company_id=2,
            thread_id="t-xyz",
        )

    assert len(captured_audit) == 1
    row = captured_audit[0]
    assert row.user_id == "tenant:2:cfo"
    assert row.action == "sql_rejected"
    assert row.severity == "warning"
