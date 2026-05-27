"""
modules/embedder.py — ONNX-accelerated BGE-M3 embeddings via fastembed.

~3x faster than PyTorch on CPU. Produces dense (1024-dim) + sparse
in one pass, same as the GPU version.
"""

import numpy as np
from config import EMBED_MODEL_ID, EMBED_BATCH_SIZE, SPARSE_TOP_K


class Embedder:
    def __init__(self, model_id: str = EMBED_MODEL_ID):
        self.model_id = model_id
        self._dense_model = None
        self._sparse_model = None

    def _load(self):
        if self._dense_model is not None:
            return
        try:
            from fastembed import TextEmbedding, SparseTextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed not installed.\n"
                "Run: pip install fastembed"
            )

        self._dense_model = TextEmbedding(
            self.model_id, providers=["CPUExecutionProvider"]
        )

        # BGE-M3 sparse may not be available in all fastembed versions
        supported = [m["model"] for m in SparseTextEmbedding.list_supported_models()]
        if self.model_id in supported:
            self._sparse_model = SparseTextEmbedding(
                self.model_id, providers=["CPUExecutionProvider"]
            )
        else:
            self._sparse_model = SparseTextEmbedding(
                "Qdrant/bm25", providers=["CPUExecutionProvider"]
            )

    def offload(self):
        pass

    def embed(self, texts: list[str]) -> dict:
        self._load()
        dense = np.array(
            list(self._dense_model.embed(texts, batch_size=EMBED_BATCH_SIZE)),
            dtype=np.float32,
        )
        sparse_raw = list(
            self._sparse_model.embed(texts, batch_size=EMBED_BATCH_SIZE)
        )
        sparse = [self._convert_sparse(s) for s in sparse_raw]
        return {"dense": dense, "sparse": sparse}

    def embed_query(self, text: str) -> dict:
        result = self.embed([text])
        return {
            "dense":  result["dense"],
            "sparse": result["sparse"][0],
        }

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-10)
        d = doc_vecs  / (np.linalg.norm(doc_vecs,  axis=1, keepdims=True) + 1e-10)
        return (q @ d.T).flatten()

    @staticmethod
    def sparse_score(query_sparse: dict, doc_sparse: dict) -> float:
        return sum(
            query_sparse.get(tok, 0.0) * weight
            for tok, weight in doc_sparse.items()
        )

    @staticmethod
    def hybrid_score(
        dense_score: float,
        sparse_score: float,
        alpha: float = 0.7,
    ) -> float:
        return alpha * dense_score + (1 - alpha) * sparse_score

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _convert_sparse(sparse_embedding) -> dict:
        """Convert fastembed SparseEmbedding to {token_id_str: weight} dict."""
        items = sorted(
            zip(sparse_embedding.indices.tolist(), sparse_embedding.values.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return {str(idx): val for idx, val in items[:SPARSE_TOP_K]}

    @staticmethod
    def _top_sparse(weights: dict, top_k: int) -> dict:
        if len(weights) <= top_k:
            return dict(weights)
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:top_k])
