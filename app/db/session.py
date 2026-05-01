"""Async SQLAlchemy engine + sessionmaker factories.

Both factories are lru-cached so they behave like singletons inside the
process while still being easy to override in tests (clear the cache and
re-import). The pattern mirrors ``app.config.get_settings``.
"""

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine bound to the application database."""
    settings = get_settings()
    return create_async_engine(
        settings.app_database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async sessionmaker."""
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        autoflush=False,
    )
