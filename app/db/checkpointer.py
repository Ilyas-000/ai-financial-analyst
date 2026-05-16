"""LangGraph PostgresSaver wiring.

``AsyncPostgresSaver`` runs through psycopg (not asyncpg) and needs a single
long-lived connection pool kept open for the lifetime of the application.
The pool is created in the FastAPI ``lifespan`` and closed on shutdown;
this module owns the construction details so the FastAPI app and the CLI
debug script can share the same setup.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings, get_settings


def _build_pool(settings: Settings) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=settings.psycopg_dsn,
        min_size=settings.checkpoint_pool_min_size,
        max_size=settings.checkpoint_pool_max_size,
        open=False,
        kwargs={
            # AsyncPostgresSaver issues DDL (setup) and writes per checkpoint;
            # autocommit avoids implicit transactions around each statement.
            "autocommit": True,
            # dict_row is required by langgraph-checkpoint-postgres internals.
            "row_factory": dict_row,
            # Prepared statements are reused across checkpoints; threshold 0
            # forces psycopg to prepare on first use which matches the saver's
            # access pattern.
            "prepare_threshold": 0,
        },
    )


@asynccontextmanager
async def checkpointer_lifespan(
    settings: Settings | None = None,
) -> AsyncIterator[tuple[AsyncConnectionPool, AsyncPostgresSaver]]:
    """Open pool, run idempotent ``setup``, yield ``(pool, saver)``.

    Used both by FastAPI ``lifespan`` and by ``scripts/chat.py`` — anything
    that needs a graph compiled with the real checkpointer.
    """
    cfg = settings or get_settings()
    pool = _build_pool(cfg)
    await pool.open(wait=True, timeout=cfg.checkpoint_pool_open_timeout)
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield pool, saver
    finally:
        await pool.close()
