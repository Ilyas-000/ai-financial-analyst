"""Idempotent ingestion of the markdown corpus into Qdrant.

Pipeline per ``make ingest`` run:
1. Walk ``docs/**/*.md`` and parse YAML frontmatter + body.
2. Compute a sha256 over the raw file bytes (frontmatter + body).
3. Diff against ``ingestion_state`` in Postgres:
   * unseen ``doc_id``                 -> embed + upsert + insert state
   * known ``doc_id`` with new hash    -> delete old chunks + embed + upsert + update state
   * known ``doc_id`` with same hash   -> skip
4. Any ``doc_id`` left in state but missing from disk -> delete chunks + remove state row.

The script is sync at the orchestration layer (Qdrant client is sync) but uses
the async SQLAlchemy session for parity with the rest of the app.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
from qdrant_client import QdrantClient
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import Settings, get_settings
from app.db.models import IngestionState
from app.db.session import get_sessionmaker
from app.rag.chunker import chunk_document
from app.rag.embedder import get_embedder
from app.rag.qdrant_store import (
    build_point,
    count_points,
    delete_doc_chunks,
    ensure_collection,
    get_qdrant_client,
    upsert_chunks,
)

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"


@dataclass(frozen=True)
class ParsedDoc:
    doc_id: str
    title: str
    version: int
    tenant_scope: str
    access_roles: tuple[str, ...]
    lang: str
    body: str
    hash: str
    relpath: str


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_doc(path: Path, docs_root: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    meta = post.metadata
    required = ("doc_id", "title", "tenant_scope")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
            f"{path.relative_to(docs_root)}: missing frontmatter keys: {missing}"
        )
    return ParsedDoc(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        version=int(meta.get("version", 1)),
        tenant_scope=str(meta["tenant_scope"]),
        access_roles=tuple(str(r) for r in meta.get("access_roles", [])),
        lang=str(meta.get("lang", "ru")),
        body=post.content,
        hash=_hash_text(raw),
        relpath=str(path.relative_to(docs_root)),
    )


def _scan_docs(docs_root: Path) -> list[ParsedDoc]:
    if not docs_root.exists():
        raise FileNotFoundError(f"docs root not found: {docs_root}")
    docs: list[ParsedDoc] = []
    seen: dict[str, str] = {}
    for path in sorted(docs_root.rglob("*.md")):
        doc = _parse_doc(path, docs_root)
        if doc.doc_id in seen:
            raise ValueError(
                f"duplicate doc_id '{doc.doc_id}': {seen[doc.doc_id]} and {doc.relpath}"
            )
        seen[doc.doc_id] = doc.relpath
        docs.append(doc)
    return docs


def _index_doc(
    doc: ParsedDoc,
    qdrant: QdrantClient,
    collection: str,
    indexed_at: datetime,
) -> int:
    chunks = chunk_document(doc.body, default_section=doc.title)
    if not chunks:
        return 0
    enc = get_embedder().encode([c.text for c in chunks])
    points = [
        build_point(
            doc_id=doc.doc_id,
            chunk_index=c.chunk_index,
            payload={
                "doc_id": doc.doc_id,
                "doc_title": doc.title,
                "doc_version": doc.version,
                "doc_hash": doc.hash,
                "section": c.section,
                "lang": doc.lang,
                "tenant_scope": doc.tenant_scope,
                "access_roles": list(doc.access_roles),
                "indexed_at": indexed_at.isoformat(),
                "text": c.text,
            },
            dense_vec=enc.dense[i],
            sparse_vec=enc.sparse[i],
        )
        for i, c in enumerate(chunks)
    ]
    upsert_chunks(qdrant, collection, points)
    return len(chunks)


async def _load_state(session_maker) -> dict[str, IngestionState]:
    async with session_maker() as session:
        result = await session.execute(select(IngestionState))
        return {row.doc_id: row for row in result.scalars()}


async def _upsert_state(
    session_maker, doc_id: str, hash_: str, version: int, indexed_at: datetime
) -> None:
    async with session_maker() as session:
        stmt = (
            pg_insert(IngestionState)
            .values(
                doc_id=doc_id, hash=hash_, version=version, indexed_at=indexed_at
            )
            .on_conflict_do_update(
                index_elements=["doc_id"],
                set_={"hash": hash_, "version": version, "indexed_at": indexed_at},
            )
        )
        await session.execute(stmt)
        await session.commit()


async def _delete_state(session_maker, doc_id: str) -> None:
    async with session_maker() as session:
        await session.execute(
            delete(IngestionState).where(IngestionState.doc_id == doc_id)
        )
        await session.commit()


async def run_ingest(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    docs = _scan_docs(DOCS_ROOT)

    qdrant = get_qdrant_client()
    ensure_collection(qdrant, settings.qdrant_collection)

    session_maker = get_sessionmaker()
    existing = await _load_state(session_maker)

    indexed_at = datetime.now(UTC)
    counters = {"indexed": 0, "re-indexed": 0, "skipped": 0, "deleted": 0, "chunks": 0}
    seen: set[str] = set()

    for doc in docs:
        seen.add(doc.doc_id)
        prev = existing.get(doc.doc_id)
        if prev is None:
            n = _index_doc(doc, qdrant, settings.qdrant_collection, indexed_at)
            await _upsert_state(session_maker, doc.doc_id, doc.hash, doc.version, indexed_at)
            counters["indexed"] += 1
            counters["chunks"] += n
            print(f"  indexed:    {doc.doc_id}  ({n} chunks)  [{doc.relpath}]")
        elif prev.hash != doc.hash:
            delete_doc_chunks(qdrant, settings.qdrant_collection, doc.doc_id)
            n = _index_doc(doc, qdrant, settings.qdrant_collection, indexed_at)
            await _upsert_state(session_maker, doc.doc_id, doc.hash, doc.version, indexed_at)
            counters["re-indexed"] += 1
            counters["chunks"] += n
            print(f"  re-indexed: {doc.doc_id}  ({n} chunks)  [{doc.relpath}]")
        else:
            counters["skipped"] += 1
            print(f"  skipped:    {doc.doc_id}  [{doc.relpath}]")

    for doc_id in existing:
        if doc_id not in seen:
            delete_doc_chunks(qdrant, settings.qdrant_collection, doc_id)
            await _delete_state(session_maker, doc_id)
            counters["deleted"] += 1
            print(f"  deleted:    {doc_id}")

    total_in_qdrant = count_points(qdrant, settings.qdrant_collection)
    print()
    print("Summary:")
    print(f"  indexed:    {counters['indexed']}")
    print(f"  re-indexed: {counters['re-indexed']}")
    print(f"  skipped:    {counters['skipped']}")
    print(f"  deleted:    {counters['deleted']}")
    print(f"  chunks now in qdrant: {total_in_qdrant}")
    return counters


def main() -> None:
    asyncio.run(run_ingest())


if __name__ == "__main__":
    main()
