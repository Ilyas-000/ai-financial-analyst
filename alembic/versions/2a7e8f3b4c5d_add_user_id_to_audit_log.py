"""add_user_id_to_audit_log

Revision ID: 2a7e8f3b4c5d
Revises: 8b32aa108f5c
Create Date: 2026-05-28 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "2a7e8f3b4c5d"
down_revision: str | None = "8b32aa108f5c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Demo identity is synthetic (``tenant:{company_id}:{role}``); pre-existing
    # rows predate this column so we backfill them with ``'legacy'`` via
    # ``server_default`` and then drop the default so future inserts must
    # provide the value explicitly.
    op.add_column(
        "audit_log",
        sa.Column(
            "user_id",
            sa.String(length=100),
            nullable=False,
            server_default=sa.text("'legacy'"),
            comment=(
                "synthetic identity of the user that issued the action; in demo "
                "shaped as 'tenant:{company_id}:{role}'."
            ),
        ),
    )
    op.alter_column("audit_log", "user_id", server_default=None)


def downgrade() -> None:
    op.drop_column("audit_log", "user_id")
