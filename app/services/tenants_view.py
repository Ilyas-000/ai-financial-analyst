"""Tenant catalogue exposed to the UI layer.

Chainlit (and any other UI we might add) needs the list of tenants to render a
tenant picker. Per ``rules.md`` §2.1 the UI layer must NOT import from
``app.db`` directly, so this service is the single point of contact between
the UI and the database for tenant metadata.

The companies table is the source of truth — same data the seed produces and
the SQL guard / Qdrant payload-filter rely on.
"""

from dataclasses import dataclass

from sqlalchemy import select

from app.db.models import Company
from app.db.session import get_sessionmaker
from app.rag.tenants import slug_from_name


@dataclass(frozen=True)
class TenantOption:
    company_id: int
    name: str
    slug: str


async def list_tenants_for_ui() -> list[TenantOption]:
    """Return all partner companies, ordered by ``id`` for stable UI rendering."""
    sm = get_sessionmaker()
    async with sm() as session:
        rows = await session.execute(select(Company.id, Company.name).order_by(Company.id))
        return [
            TenantOption(company_id=row.id, name=row.name, slug=slug_from_name(row.name))
            for row in rows
        ]
