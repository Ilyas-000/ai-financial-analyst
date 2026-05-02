"""Hybrid retrieval: dense + sparse via Qdrant native RRF.

Both branches share a payload filter that enforces tenant isolation
(``tenant_scope`` ∈ {``tenant:*``, ``tenant:<slug>``}) and role access
(``access_roles`` list contains the caller's role) at retrieval time —
not as a post-filter.

The RRF fusion is performed by Qdrant itself (default ``k=60``), so we only
size the per-branch prefetches and the post-fusion limit. Score is replaced
by the cross-encoder later when rerank is enabled.
"""

import asyncio
from dataclasses import dataclass

from qdrant_client.http import models as qm

from app.config import Settings, get_settings
from app.rag.embedder import get_embedder
from app.rag.qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    text: str
    doc_id: str
    doc_title: str
    section: str | None
    tenant_scope: str
    access_roles: tuple[str, ...]


def _payload_filter(tenant_slug: str, user_role: str) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="tenant_scope",
                match=qm.MatchAny(any=["tenant:*", f"tenant:{tenant_slug}"]),
            ),
            qm.FieldCondition(
                key="access_roles",
                match=qm.MatchValue(value=user_role),
            ),
        ],
    )


def _to_chunk(point) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        score=float(point.score) if point.score is not None else 0.0,
        text=str(payload.get("text", "")),
        doc_id=str(payload.get("doc_id", "")),
        doc_title=str(payload.get("doc_title", "")),
        section=payload.get("section"),
        tenant_scope=str(payload.get("tenant_scope", "")),
        access_roles=tuple(payload.get("access_roles", [])),
    )


def _query_sync(
    collection: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    flt: qm.Filter,
    top_k_dense: int,
    top_k_sparse: int,
    out_top_k: int,
) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    sparse_q = qm.SparseVector(
        indices=list(sparse_vec.keys()),
        values=list(sparse_vec.values()),
    )
    response = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=top_k_dense,
                filter=flt,
            ),
            qm.Prefetch(
                query=sparse_q,
                using=SPARSE_VECTOR_NAME,
                limit=top_k_sparse,
                filter=flt,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=flt,
        limit=out_top_k,
        with_payload=True,
    )
    return [_to_chunk(p) for p in response.points]


async def hybrid_search(
    question: str,
    user_role: str,
    tenant_slug: str,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``rerank_input_top_k`` chunks ranked by Qdrant RRF."""
    s = settings or get_settings()
    encoded = get_embedder().encode([question])
    flt = _payload_filter(tenant_slug, user_role)
    return await asyncio.to_thread(
        _query_sync,
        s.qdrant_collection,
        encoded.dense[0],
        encoded.sparse[0],
        flt,
        s.retrieval_top_k_dense,
        s.retrieval_top_k_sparse,
        s.rerank_input_top_k,
    )
