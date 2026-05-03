"""Single entry point for guarded SQL execution + audit logging.

``execute_guarded(sql, user_role, company_id, ...)`` is the only function the
SQL Analyst subgraph calls. It chains:

  1. ``guard_sql(...)`` — guard rejects → audit ``sql_rejected`` and re-raise.
  2. ``SET LOCAL statement_timeout`` + ``session.execute(text(sql))`` inside a
     dedicated transaction — DB error → audit ``sql_failed`` and raise
     ``SQLExecutionError`` so the ReAct loop in the subgraph can self-correct.
  3. Successful run → audit with ``guard_result.audit_action`` (either
     ``sql_executed`` or ``tenancy_rewrite``) and return ``ExecutionResult``.

Audit writes go through a fresh session so they survive a query rollback.
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db.models import AuditLog
from app.db.session import get_sessionmaker
from app.tools.sql_guard import GuardRejection, GuardResult, guard_sql


@dataclass(frozen=True)
class ExecutionResult:
    guard: GuardResult
    rows: list[dict[str, Any]]
    rows_returned: int
    elapsed_ms: int


class SQLExecutionError(Exception):
    """Raised when guard passed but the database rejected the SQL.

    Carries the normalized SQL so the ReAct loop can show it back to the LLM.
    """

    def __init__(
        self,
        message: str,
        *,
        guard_result: GuardResult,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.guard_result = guard_result
        self.original = original


async def execute_guarded(
    sql: str,
    *,
    user_role: str,
    company_id: int,
    thread_id: str | None = None,
) -> ExecutionResult:
    """Guard, execute, audit. The only public function in this module."""
    settings = get_settings()
    started = time.monotonic()

    try:
        guard_result = guard_sql(sql, user_role=user_role, company_id=company_id)
    except GuardRejection as exc:
        await _write_audit(
            company_id=company_id,
            user_role=user_role,
            action="sql_rejected",
            severity=exc.severity,
            sql_text=sql,
            rows_returned=None,
            details={**exc.details, "raw_sql": sql, "thread_id": thread_id},
        )
        raise

    try:
        rows = await _run_query(guard_result.sql, settings.sql_statement_timeout_ms)
    except SQLAlchemyError as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        message = _short_db_error(exc)
        await _write_audit(
            company_id=company_id,
            user_role=user_role,
            action="sql_failed",
            severity="warning",
            sql_text=guard_result.sql,
            rows_returned=None,
            details={
                **guard_result.audit_details,
                "error": message,
                "elapsed_ms": elapsed_ms,
                "raw_sql": sql,
                "thread_id": thread_id,
            },
        )
        raise SQLExecutionError(message, guard_result=guard_result, original=exc) from exc

    elapsed_ms = int((time.monotonic() - started) * 1000)
    await _write_audit(
        company_id=company_id,
        user_role=user_role,
        action=guard_result.audit_action,
        severity=guard_result.audit_severity,
        sql_text=guard_result.sql,
        rows_returned=len(rows),
        details={
            **guard_result.audit_details,
            "elapsed_ms": elapsed_ms,
            "raw_sql": sql,
            "thread_id": thread_id,
        },
    )

    return ExecutionResult(
        guard=guard_result,
        rows=rows,
        rows_returned=len(rows),
        elapsed_ms=elapsed_ms,
    )


async def _run_query(sql: str, statement_timeout_ms: int) -> list[dict[str, Any]]:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session, session.begin():
        # SET LOCAL needs an active transaction (which `session.begin()` provides).
        # Casting to int prevents string injection through the env-driven setting.
        await session.execute(text(f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}"))
        result = await session.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]


async def _write_audit(
    *,
    company_id: int,
    user_role: str,
    action: str,
    severity: str,
    sql_text: str | None,
    rows_returned: int | None,
    details: dict[str, Any] | None,
) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session, session.begin():
        session.add(
            AuditLog(
                company_id=company_id,
                user_role=user_role,
                action=action,
                sql_text=sql_text,
                rows_returned=rows_returned,
                severity=severity,
                details=details,
                created_at=datetime.now(UTC),
            )
        )


def _short_db_error(exc: SQLAlchemyError) -> str:
    """Strip multiline driver dumps so the LLM can read the error."""
    raw = str(getattr(exc, "orig", None) or exc)
    first_line = raw.strip().splitlines()[0] if raw.strip() else exc.__class__.__name__
    return first_line[:500]
