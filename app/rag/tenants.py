"""Map ``company_id`` to the tenant slug used in Qdrant payloads and ``docs/``.

The slug convention is the lowercased first whitespace/hyphen-separated token
of ``companies.name`` — matches the ``docs/tenants/<slug>/`` layout produced by
the seed (``ACME LLC`` → ``acme``, ``Ostrovok-mock`` → ``ostrovok``,
``CheckScan-mock`` → ``checkscan``).
"""

import re

from sqlalchemy import select

from app.db.models import Company
from app.db.session import get_sessionmaker

_TOKEN_SPLIT = re.compile(r"[\s\-]+")


def slug_from_name(name: str) -> str:
    parts = _TOKEN_SPLIT.split(name.strip())
    if not parts or not parts[0]:
        raise ValueError(f"cannot derive tenant slug from name: {name!r}")
    return parts[0].lower()


async def tenant_slug_for(company_id: int) -> str:
    sm = get_sessionmaker()
    async with sm() as session:
        row = await session.execute(
            select(Company.name).where(Company.id == company_id)
        )
        name = row.scalar_one_or_none()
    if name is None:
        raise ValueError(f"unknown company_id={company_id}")
    return slug_from_name(name)
