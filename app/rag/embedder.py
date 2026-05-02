"""BGE-M3 embedder producing dense + sparse vectors in a single forward pass.

The model is heavy (~1.3 GiB FP16) so it is loaded lazily on the first
``encode()`` call and kept in process memory afterwards. The factory
``get_embedder()`` is ``lru_cache``'d to keep the singleton boundary explicit.

Dense vectors are 1024-d cosine-normalised by the model.
Sparse vectors are returned by FlagEmbedding as ``{token_id_str: weight}`` —
we convert keys to ``int`` to match Qdrant's ``SparseVector(indices, values)``
contract.
"""

from dataclasses import dataclass
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

from app.config import get_settings

_DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class EmbeddingResult:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class BGEM3Embedder:
    def __init__(self, model_name: str, use_fp16: bool) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: BGEM3FlagModel | None = None

    def _ensure_loaded(self) -> BGEM3FlagModel:
        if self._model is None:
            self._model = BGEM3FlagModel(self._model_name, use_fp16=self._use_fp16)
        return self._model

    def encode(
        self, texts: list[str], batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])
        model = self._ensure_loaded()
        out = model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [vec.tolist() for vec in out["dense_vecs"]]
        sparse = [
            {int(token_id): float(weight) for token_id, weight in entry.items()}
            for entry in out["lexical_weights"]
        ]
        return EmbeddingResult(dense=dense, sparse=sparse)


@lru_cache(maxsize=1)
def get_embedder() -> BGEM3Embedder:
    s = get_settings()
    return BGEM3Embedder(model_name=s.embedding_model, use_fp16=True)
