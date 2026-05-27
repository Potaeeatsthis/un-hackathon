"""
modules/reranker.py — ONNX-accelerated BGE reranker via fastembed.

~3x faster than PyTorch on CPU. Same RankedHit interface as GPU version.
Scores are sigmoid-normalized to [0,1] for threshold compatibility.
"""

import math
from dataclasses import dataclass
from config import RERANKER_MODEL_ID, RERANK_TOP_N, RERANK_THRESHOLD
from modules.faiss_index import Hit


@dataclass
class RankedHit:
    hit: Hit
    reranker_score: float

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


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class Reranker:
    def __init__(self, model_id: str = RERANKER_MODEL_ID):
        self.model_id = model_id
        self._model   = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(
                model_name=self.model_id,
                providers=["CPUExecutionProvider"],
            )
        except ImportError:
            raise ImportError(
                "fastembed not installed.\n"
                "Run: pip install fastembed"
            )

    def offload(self):
        pass

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        top_n: int = RERANK_TOP_N,
    ) -> list[RankedHit]:
        if not hits:
            return []

        self._load()
        documents = [h.text for h in hits]
        raw_scores = list(self._model.rerank(query, documents))

        ranked = []
        for h, score in zip(hits, raw_scores):
            s = float(score) if isinstance(score, (int, float)) else float(score.score)
            ranked.append(RankedHit(hit=h, reranker_score=_sigmoid(s)))

        ranked.sort(key=lambda r: r.reranker_score, reverse=True)
        return ranked[:top_n]

    @staticmethod
    def needs_fallback(ranked: list["RankedHit"]) -> bool:
        if not ranked:
            return True
        return ranked[0].reranker_score < RERANK_THRESHOLD
