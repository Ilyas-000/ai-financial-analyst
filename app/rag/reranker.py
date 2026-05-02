"""Cross-encoder reranker (BGE-reranker-v2-m3) over hybrid-retrieval candidates.

Loaded lazily and held as a process-local singleton — symmetric to the BGE-M3
embedder. FP16 weights are requested via ``model_kwargs={"torch_dtype": ...}``
to halve memory; we silently fall back to FP32 when the underlying device or
HF revision rejects FP16.

Gated by ``settings.enable_rerank``: when disabled (e.g. local dev with low
RAM), callers receive the RRF top-N pass-through; the cross-encoder model is
never loaded.
"""

import asyncio
from dataclasses import replace
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.rag.retriever import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, use_fp16: bool) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        kwargs: dict = {}
        if self._use_fp16:
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        try:
            self._model = CrossEncoder(self._model_name, **kwargs)
        except (TypeError, ValueError, RuntimeError):
            # FP16 unsupported on the active device / HF revision — fall back.
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._ensure_loaded()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_k]]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    s = get_settings()
    return CrossEncoderReranker(model_name=s.reranker_model, use_fp16=True)


async def rerank_async(
    query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    return await asyncio.to_thread(get_reranker().rerank, query, candidates, top_k)
