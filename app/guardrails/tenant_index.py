"""Lazy tenant catalogue used by the output guard.

The output guard needs to know every other tenant's display name and INN to
detect cross-tenant leaks in the final answer. Loaded once per process via an
async cache — the tenant list is stable during a deployment (new tenants land
through seed/migrations, not the API), so a single trip to Postgres is enough.

Kept separate from :mod:`app.services.tenants_view` because that service is
UI-facing (returns slugs for the Chainlit picker, no INN); mixing concerns
would force the UI to load secrets it doesn't display.
"""

import asyncio
from dataclasses import dataclass

from sqlalchemy import select

from app.db.models import Company
from app.db.session import get_sessionmaker


@dataclass(frozen=True)
class TenantEntry:
    company_id: int
    name: str
    inn: str


_cache: list[TenantEntry] | None = None
_cache_lock = asyncio.Lock()


async def load_tenant_index() -> list[TenantEntry]:
    """Return all tenants, cached for the process lifetime.

    Concurrency-safe: ``asyncio.Lock`` serialises the first concurrent load so
    we don't issue duplicate SELECTs from parallel chat requests.
    """
    global _cache
    if _cache is not None:
        return _cache
    async with _cache_lock:
        if _cache is not None:
            return _cache
        sm = get_sessionmaker()
        async with sm() as session:
            rows = await session.execute(
                select(Company.id, Company.name, Company.inn).order_by(Company.id)
            )
            _cache = [TenantEntry(company_id=row.id, name=row.name, inn=row.inn) for row in rows]
        return _cache


def reset_tenant_index_cache() -> None:
    """Test hook to drop the cached index between runs."""
    global _cache
    _cache = None
