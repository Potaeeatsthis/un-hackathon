"""
modules/faiss_index.py — FAISS index management.

Two logical indexes:
  main  — permanent country law indexes (faiss:{country} in cache)
  temp  — per-session upload index     (faiss:temp in cache, TTL 72h)

Both are flat L2 indexes for the prototype (swap to IVF for >10k vecs).
Dense vectors only for FAISS; sparse scoring applied as a post-pass.

Usage:
    from modules.faiss_index import IndexManager
    idx = IndexManager()

    # Add segments + their embeddings
    idx.add(segments, embeddings, index="temp")

    # Search returns Hit dicts sorted by hybrid score
    hits = idx.search(query_emb, top_k=20, index="both")
"""

import io
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from config import FAISS_TOP_K_RETRIEVE, TTL_TEMP_INDEX, TTL_COUNTRY_INDEX

# Import FAISS (CPU or GPU)
try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed.\nRun: pip install faiss-cpu")

from modules.cache import cache
from modules.embedder import Embedder


IndexName = Literal["main", "temp", "both"]
Country   = str   # e.g. "TH", "VN", "ID"


@dataclass
class Hit:
    section_id: str
    text: str
    page: int
    doc_name: str
    country: str
    dense_score: float
    sparse_score: float = 0.0
    hybrid_score: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


class _SingleIndex:
    """
    Wraps one FAISS flat index + a parallel list of segment metadata.
    Stored as (faiss_bytes, segments_list) in cache.
    """

    def __init__(self, dim: int = 1024):
        self.dim  = dim
        self._idx = faiss.IndexFlatIP(dim)   # inner product (after L2-norm = cosine)
        self._segments: list[dict] = []
        self._sparse: list[dict]   = []       # parallel sparse weights

    def add(self, segments: list[dict], dense: np.ndarray, sparse: list[dict]):
        # L2-normalise for cosine via inner product
        faiss.normalize_L2(dense)
        self._idx.add(dense)
        self._segments.extend(segments)
        self._sparse.extend(sparse)

    def search(
        self,
        query_dense: np.ndarray,
        query_sparse: dict,
        top_k: int,
        alpha: float = 0.7,
    ) -> list[Hit]:
        if self._idx.ntotal == 0:
            return []

        q = query_dense.copy()
        faiss.normalize_L2(q)

        k = min(top_k, self._idx.ntotal)
        scores, ids = self._idx.search(q, k)

        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            seg = self._segments[idx]
            sp  = self._sparse[idx]
            s_score = Embedder.sparse_score(query_sparse, sp)
            h_score = Embedder.hybrid_score(float(score), s_score, alpha)
            hits.append(Hit(
                section_id   = seg["section_id"],
                text         = seg["text"],
                page         = seg["page"],
                doc_name     = seg["doc_name"],
                country      = seg["country"],
                dense_score  = float(score),
                sparse_score = s_score,
                hybrid_score = h_score,
            ))
        return hits

    def ntotal(self) -> int:
        return self._idx.ntotal

    # ── Serialise to / from bytes (for cache storage) ──────────────────────

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        faiss.write_index(self._idx, faiss.PyCallbackIOWriter(buf.write))
        return pickle.dumps({
            "faiss": buf.getvalue(),
            "segments": self._segments,
            "sparse": self._sparse,
            "dim": self.dim,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> "_SingleIndex":
        payload = pickle.loads(data)
        obj = cls(dim=payload["dim"])
        buf = io.BytesIO(payload["faiss"])
        obj._idx      = faiss.read_index(faiss.PyCallbackIOReader(buf.read))
        obj._segments = payload["segments"]
        obj._sparse   = payload["sparse"]
        return obj


class IndexManager:
    """
    Manages temp index (in RAM) and main country indexes (loaded from cache).
    """

    def __init__(self, dim: int = 1024):
        self.dim   = dim
        self._temp: Optional[_SingleIndex] = None
        self._main: dict[Country, _SingleIndex] = {}   # country → index

    # ── Add ────────────────────────────────────────────────────────────────────

    def add(
        self,
        segments: list[dict],
        embeddings: dict,                 # {"dense": np.ndarray, "sparse": list[dict]}
        index: Literal["temp", "main"],
        country: Country = "XX",
    ):
        dense  = embeddings["dense"]
        sparse = embeddings["sparse"]

        if index == "temp":
            if self._temp is None:
                self._temp = _SingleIndex(self.dim)
            self._temp.add(segments, dense, sparse)
            # Persist to cache
            cache.set("faiss:temp", self._temp.to_bytes(), ttl=TTL_TEMP_INDEX)

        elif index == "main":
            if country not in self._main:
                self._main[country] = _SingleIndex(self.dim)
            self._main[country].add(segments, dense, sparse)
            cache.set(f"faiss:{country}", self._main[country].to_bytes(), ttl=TTL_COUNTRY_INDEX)
            cache.set(f"loaded:{country}", True, ttl=TTL_COUNTRY_INDEX)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_emb: dict,           # {"dense": (1,D), "sparse": dict}
        top_k: int = FAISS_TOP_K_RETRIEVE,
        index: IndexName = "both",
        country: Optional[Country] = None,
        alpha: float = 0.7,
    ) -> list[Hit]:
        """
        Returns hits sorted by hybrid_score descending.
        index="both"   → search temp + all loaded main indexes
        index="temp"   → temp only
        index="main"   → all main (or single country if country= given)
        """
        q_dense  = query_emb["dense"]
        q_sparse = query_emb["sparse"]
        hits: list[Hit] = []

        if index in ("both", "temp"):
            self._ensure_temp()
            if self._temp:
                hits += self._temp.search(q_dense, q_sparse, top_k, alpha)

        if index in ("both", "main"):
            targets = [country] if country else list(self._main.keys())
            for c in targets:
                self._ensure_main(c)
                if c in self._main:
                    hits += self._main[c].search(q_dense, q_sparse, top_k, alpha)

        hits.sort(key=lambda h: h.hybrid_score, reverse=True)
        return hits[:top_k]

    # ── Info ───────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        temp_n = self._temp.ntotal() if self._temp else 0
        main_n = {c: idx.ntotal() for c, idx in self._main.items()}
        loaded = cache.keys("loaded:*")
        return {
            "temp_vectors": temp_n,
            "main_vectors": main_n,
            "loaded_countries": [k.split(":")[1] for k in loaded],
        }

    # ── Clear ──────────────────────────────────────────────────────────────────

    def clear(self, index: Literal["temp", "main", "all"]):
        if index in ("temp", "all"):
            self._temp = None
            cache.delete("faiss:temp")
        if index in ("main", "all"):
            self._main.clear()
            for k in cache.keys("faiss:*"):
                if k != "faiss:temp":
                    cache.delete(k)
            for k in cache.keys("loaded:*"):
                cache.delete(k)

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _ensure_temp(self):
        if self._temp is None:
            raw = cache.get("faiss:temp")
            if raw:
                self._temp = _SingleIndex.from_bytes(raw)

    def _ensure_main(self, country: Country):
        if country not in self._main:
            raw = cache.get(f"faiss:{country}")
            if raw:
                self._main[country] = _SingleIndex.from_bytes(raw)
