"""Render a role-scoped textual schema for the SQL Analyst prompt.

Walks ``app.db.models.Base.metadata``, keeps only tables the role can access
per ``ROLE_TABLE_ALLOWLIST``, and hides sensitive columns from non-privileged
roles. Output is a single string of ``-- comment\\nCREATE TABLE ...`` blocks
ordered alphabetically — small enough (~2 KB for v1) to drop directly into
the system prompt instead of running a schema retriever.
"""

from sqlalchemy import Column, Table
from sqlalchemy.dialects import postgresql

from app.db.models import Base
from app.tools.sql_guard import (
    ROLE_TABLE_ALLOWLIST,
    SENSITIVE_ALLOWED_ROLES,
    SENSITIVE_COLUMN_OWNERS,
)

_PG_DIALECT = postgresql.dialect()


def schema_for_role(user_role: str) -> str:
    if user_role not in ROLE_TABLE_ALLOWLIST:
        raise ValueError(f"unknown role: {user_role!r}")

    allowed_tables = ROLE_TABLE_ALLOWLIST[user_role]
    hide_sensitive = user_role not in SENSITIVE_ALLOWED_ROLES
    table_to_hidden_columns = _table_to_hidden_columns() if hide_sensitive else {}

    metadata = Base.metadata
    blocks: list[str] = []
    for table_name in sorted(allowed_tables):
        table = metadata.tables.get(table_name)
        if table is None:
            continue
        hidden = table_to_hidden_columns.get(table_name, frozenset())
        blocks.append(_format_table(table, hidden))

    return "\n\n".join(blocks)


def format_table_for_role(table_name: str, user_role: str) -> str | None:
    """Render a single ``-- comment\\nCREATE TABLE ...`` block for ``table_name``.

    Returns ``None`` if the table is unknown or not visible to the role; sensitive
    columns are hidden using the same rules as ``schema_for_role``.
    """
    if user_role not in ROLE_TABLE_ALLOWLIST:
        return None
    if table_name not in ROLE_TABLE_ALLOWLIST[user_role]:
        return None
    table = Base.metadata.tables.get(table_name)
    if table is None:
        return None
    hidden = frozenset()
    if user_role not in SENSITIVE_ALLOWED_ROLES:
        hidden = _table_to_hidden_columns().get(table_name, frozenset())
    return _format_table(table, hidden)


def tables_with_column(column_name: str, user_role: str) -> list[str]:
    """Return tables visible to ``user_role`` that have a column ``column_name``.

    Sensitive columns hidden for the role are also excluded — if the LLM can't
    see the column in its schema view, telling it "X exists in Y" would be
    misleading.
    """
    if user_role not in ROLE_TABLE_ALLOWLIST:
        return []
    allowed = ROLE_TABLE_ALLOWLIST[user_role]
    hidden_map = (
        _table_to_hidden_columns()
        if user_role not in SENSITIVE_ALLOWED_ROLES
        else {}
    )
    matches: list[str] = []
    for table_name in sorted(allowed):
        table = Base.metadata.tables.get(table_name)
        if table is None:
            continue
        hidden = hidden_map.get(table_name, frozenset())
        if column_name in table.columns and column_name not in hidden:
            matches.append(table_name)
    return matches


def _table_to_hidden_columns() -> dict[str, frozenset[str]]:
    inverted: dict[str, set[str]] = {}
    for col_name, owners in SENSITIVE_COLUMN_OWNERS.items():
        for owner in owners:
            inverted.setdefault(owner, set()).add(col_name)
    return {table: frozenset(cols) for table, cols in inverted.items()}


def _format_table(table: Table, hidden_columns: frozenset[str]) -> str:
    header = f"-- {table.name}"
    if table.comment:
        header += f": {table.comment.strip()}"

    visible_columns = [c for c in table.columns if c.name not in hidden_columns]
    column_lines = [
        _format_column(column, is_last=(index == len(visible_columns) - 1))
        for index, column in enumerate(visible_columns)
    ]
    body = "\n".join(column_lines)
    return f"{header}\nCREATE TABLE {table.name} (\n{body}\n);"


def _format_column(column: Column, *, is_last: bool) -> str:
    type_str = column.type.compile(dialect=_PG_DIALECT)
    nullable = "" if column.nullable else " NOT NULL"
    suffix = "" if is_last else ","
    base = f"  {column.name} {type_str}{nullable}{suffix}"
    if column.comment:
        base += f"  -- {column.comment.strip()}"
    return base
