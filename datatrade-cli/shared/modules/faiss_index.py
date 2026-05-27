"""
modules/faiss_index.py — FAISS index management.

Two logical indexes:
  main  — permanent country law indexes (faiss:{country} in cache)
  temp  — per-session upload index     (faiss:temp in cache, TTL 72h)

Both are flat L2 indexes for the prototype (swap to IVF for >10k vecs).
Dense vectors only for FAISS; sparse scoring applied as a post-pass.
"""

import io
import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from config import FAISS_TOP_K_RETRIEVE, TTL_TEMP_INDEX, TTL_COUNTRY_INDEX, CACHE_DIR, COUNTRY_SOURCES

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed.\nRun: pip install faiss-cpu")

from modules.cache import cache
from modules.embedder import Embedder


IndexName  = Literal["main", "temp", "both"]
Country    = str
EMBED_DIM  = 1024


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
    def __init__(self, dim: int = 1024):
        self.dim  = dim
        self._idx = faiss.IndexFlatIP(dim)
        self._segments: list[dict] = []
        self._sparse: list[dict]   = []

    def add(self, segments: list[dict], dense: np.ndarray, sparse: list[dict]):
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
    def __init__(self, dim: int = 1024):
        self.dim   = dim
        self._temp: Optional[_SingleIndex] = None
        self._main: dict[Country, _SingleIndex] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        for country in COUNTRY_SOURCES:
            pkl_path = os.path.join(CACHE_DIR, f"faiss_{country}.pkl")
            if os.path.exists(pkl_path) and country not in self._main:
                try:
                    with open(pkl_path, "rb") as f:
                        self._main[country] = _SingleIndex.from_bytes(pickle.load(f))
                    cache.set(f"loaded:{country}", True, ttl=TTL_COUNTRY_INDEX)
                except Exception:
                    pass

    def add(
        self,
        segments: list[dict],
        embeddings: dict,
        index: Literal["temp", "main"],
        country: Country = "XX",
    ):
        dense  = embeddings["dense"]
        sparse = embeddings["sparse"]

        if index == "temp":
            if self._temp is None:
                self._temp = _SingleIndex(self.dim)
            self._temp.add(segments, dense, sparse)
            cache.set("faiss:temp", self._temp.to_bytes(), ttl=TTL_TEMP_INDEX)

        elif index == "main":
            if country not in self._main:
                self._main[country] = _SingleIndex(self.dim)
            self._main[country].add(segments, dense, sparse)
            cache.set(f"faiss:{country}", self._main[country].to_bytes(), ttl=TTL_COUNTRY_INDEX)
            cache.set(f"loaded:{country}", True, ttl=TTL_COUNTRY_INDEX)

    def search(
        self,
        query_emb: dict,
        top_k: int = FAISS_TOP_K_RETRIEVE,
        index: IndexName = "both",
        country: Optional[Country] = None,
        alpha: float = 0.7,
    ) -> list[Hit]:
        q_dense  = query_emb["dense"]
        q_sparse = query_emb["sparse"]
        hits: list[Hit] = []

        if index in ("both", "temp"):
            self._ensure_temp()
            if self._temp:
                hits += self._temp.search(q_dense, q_sparse, top_k, alpha)

        if index in ("both", "main"):
            if not country:
                self._ensure_all_main()
            targets = [country] if country else list(self._main.keys())
            for c in targets:
                self._ensure_main(c)
                if c in self._main:
                    hits += self._main[c].search(q_dense, q_sparse, top_k, alpha)

        hits.sort(key=lambda h: h.hybrid_score, reverse=True)
        return hits[:top_k]

    def info(self) -> dict:
        temp_n = self._temp.ntotal() if self._temp else 0
        main_n = {c: idx.ntotal() for c, idx in self._main.items()}
        loaded = cache.keys("loaded:*")
        return {
            "temp_vectors": temp_n,
            "main_vectors": main_n,
            "loaded_countries": [k.split(":")[1] for k in loaded],
        }

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

    def _ensure_all_main(self):
        for key in cache.keys("faiss:*"):
            if key == "faiss:temp":
                continue
            country = key.split(":", 1)[1]
            if country not in self._main:
                raw = cache.get(key)
                if raw:
                    self._main[country] = _SingleIndex.from_bytes(raw)
