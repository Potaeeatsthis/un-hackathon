"""
modules/embedder.py — BGE-M3 dense + sparse embeddings.

BGE-M3 is a single model that produces:
  • dense  — 1024-dim float32 vector  (cosine similarity)
  • sparse — {token: weight} dict      (exact keyword match, BM25-like)
  • multi-vector — skipped in this prototype (costly, see PROTOTYPE.md)

Both dense and sparse are produced in one forward pass.
The model handles multilingual input (Thai + English + more) natively.

Usage:
    from modules.embedder import Embedder
    emb = Embedder()

    # batch
    result = emb.embed(["Data controller must appoint DPO", "มาตรา 28"])
    result["dense"]   # np.ndarray  shape (2, 1024)
    result["sparse"]  # list of dicts [{"DPO": 0.92, ...}, {...}]

    # single query (normalised for search)
    q = emb.embed_query("who appoints DPO?")
    q["dense"]   # shape (1, 1024)
    q["sparse"]  # dict
"""

import numpy as np
from typing import Union
from config import DEVICE, EMBED_MODEL_ID, EMBED_BATCH_SIZE, SPARSE_TOP_K


class Embedder:
    def __init__(self, model_id: str = EMBED_MODEL_ID):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.model_id,
                use_fp16=(DEVICE != "cpu"),  # fp16 on GPU, fp32 on CPU
            )
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed.\n"
                "Run: pip install FlagEmbedding"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def offload(self):
        """Move model to CPU and free VRAM. Automatically restored on next embed call."""
        if self._model is None:
            return
        import torch
        self._model.model.to("cpu")
        torch.cuda.empty_cache()

    def _ensure_gpu(self):
        if self._model is None or DEVICE == "cpu":
            return
        import torch
        device = next(self._model.model.parameters()).device
        if str(device) == "cpu":
            self._model.model.to(DEVICE)

    def embed(self, texts: list[str]) -> dict:
        """
        Embed a batch of texts (document side).

        Returns
        -------
        {
          "dense":  np.ndarray  shape (N, 1024),
          "sparse": list[dict[str, float]]   length N,
        }
        """
        self._load()
        self._ensure_gpu()
        output = self._model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,   # multi-vector skipped
        )
        dense  = np.array(output["dense_vecs"], dtype=np.float32)
        sparse = [
            self._top_sparse(s, SPARSE_TOP_K)
            for s in output["lexical_weights"]
        ]
        return {"dense": dense, "sparse": sparse}

    def embed_query(self, text: str) -> dict:
        """
        Embed a single query string.
        Identical call signature but returns N=1 arrays.
        """
        result = self.embed([text])
        return {
            "dense":  result["dense"],            # shape (1, 1024)
            "sparse": result["sparse"][0],        # single dict
        }

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        query_vec : (1, D)
        doc_vecs  : (N, D)
        returns   : (N,) float32
        """
        q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-10)
        d = doc_vecs  / (np.linalg.norm(doc_vecs,  axis=1, keepdims=True) + 1e-10)
        return (q @ d.T).flatten()

    @staticmethod
    def sparse_score(query_sparse: dict, doc_sparse: dict) -> float:
        """Dot product over shared tokens (BM25-like)."""
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
        """Weighted combination. alpha=1.0 → dense only, 0.0 → sparse only."""
        return alpha * dense_score + (1 - alpha) * sparse_score

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _top_sparse(weights: dict, top_k: int) -> dict:
        """Keep only the top-k tokens by weight."""
        if len(weights) <= top_k:
            return dict(weights)
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:top_k])
