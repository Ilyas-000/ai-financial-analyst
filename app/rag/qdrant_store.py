"""Qdrant collection setup and chunk-level upsert / delete primitives.

Collection layout — two named vectors per point:
* ``dense``  — 1024-d cosine-normalised (BGE-M3 dense head).
* ``sparse`` — token-id → weight (BGE-M3 lexical head).

Point IDs are deterministic UUIDv5 over ``(doc_id, chunk_index)`` so re-ingest
of an unchanged document yields identical IDs and overwrites in place.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_DIM = 1024

# Stable namespace so chunk UUIDs are reproducible across runs and machines.
_CHUNK_NAMESPACE = uuid.UUID("4f1c0d2e-2f8b-4a1b-9c0c-3a8e5e0a1b00")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_http_port, timeout=60)


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}::{chunk_index}"))


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=DENSE_VECTOR_DIM, distance=qm.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(),
        },
    )


def build_point(
    doc_id: str,
    chunk_index: int,
    payload: dict,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
) -> qm.PointStruct:
    return qm.PointStruct(
        id=chunk_point_id(doc_id, chunk_index),
        vector={
            DENSE_VECTOR_NAME: dense_vec,
            SPARSE_VECTOR_NAME: qm.SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values()),
            ),
        },
        payload=payload,
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    points: list[qm.PointStruct],
) -> None:
    if not points:
        return
    client.upsert(collection_name=collection_name, points=points)


def delete_doc_chunks(
    client: QdrantClient, collection_name: str, doc_id: str
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="doc_id", match=qm.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
    )


def count_points(client: QdrantClient, collection_name: str) -> int:
    return client.count(collection_name=collection_name, exact=True).count
