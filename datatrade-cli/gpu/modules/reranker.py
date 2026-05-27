"""
modules/reranker.py — BGE-reranker-v2-m3 cross-encoder.

Takes the top-K hits from FAISS and scores each (query, chunk) pair
directly. This is the "smart brain" second-pass that fixes ranking errors
from the fast vector search.

Returns the top-N hits re-ordered by reranker score.
If top-1 score < RERANK_THRESHOLD, caller should trigger web fallback.

Usage:
    from modules.reranker import Reranker
    rrk = Reranker()
    ranked = rrk.rerank("who must appoint DPO", hits, top_n=5)
    for h in ranked:
        print(h.reranker_score, h.section_id)
"""

from dataclasses import dataclass
from typing import Optional
from config import DEVICE, RERANKER_MODEL_ID, RERANK_TOP_N, RERANK_THRESHOLD
from modules.faiss_index import Hit


@dataclass
class RankedHit:
    """Hit augmented with reranker score."""
    hit: Hit
    reranker_score: float

    # Delegate common fields for convenience
    @property
    def section_id(self):    return self.hit.section_id
    @property
    def text(self):          return self.hit.text
    @property
    def page(self):          return self.hit.page
    @property
    def doc_name(self):      return self.hit.doc_name
    @property
    def country(self):       return self.hit.country
    @property
    def dense_score(self):   return self.hit.dense_score
    @property
    def hybrid_score(self):  return self.hit.hybrid_score

    def to_dict(self) -> dict:
        return {**self.hit.to_dict(), "reranker_score": self.reranker_score}


class Reranker:
    def __init__(self, model_id: str = RERANKER_MODEL_ID):
        self.model_id = model_id
        self._model   = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_id,
                use_fp16=(DEVICE != "cpu"),
            )
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed.\n"
                "Run: pip install FlagEmbedding"
            )

    def offload(self):
        """Move model to CPU and free VRAM. Automatically restored on next rerank call."""
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

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        top_n: int = RERANK_TOP_N,
    ) -> list[RankedHit]:
        """
        Parameters
        ----------
        query  : raw user question string
        hits   : output of IndexManager.search()
        top_n  : how many to return

        Returns
        -------
        list[RankedHit] sorted by reranker_score descending
        """
        if not hits:
            return []

        self._load()
        self._ensure_gpu()
        pairs = [[query, h.text] for h in hits]
        scores = self._model.compute_score(pairs, normalize=True)

        ranked = [
            RankedHit(hit=h, reranker_score=float(s))
            for h, s in zip(hits, scores)
        ]
        ranked.sort(key=lambda r: r.reranker_score, reverse=True)
        return ranked[:top_n]

    @staticmethod
    def needs_fallback(ranked: list[RankedHit]) -> bool:
        """True if the best chunk doesn't meet the confidence threshold."""
        if not ranked:
            return True
        return ranked[0].reranker_score < RERANK_THRESHOLD
