"""
modules/pipeline.py — CPU-optimized RAG query flow.

Changes from GPU version:
  - Skip reranker when FAISS scores are decisive (saves 5-10s)
  - No GPU offload/restore cycles
  - All models use ONNX (embedder, reranker) or llama.cpp (LLM)
"""

import hashlib
import time
from typing import Generator, Optional
from config import (
    TTL_QUERY_CACHE, RERANK_TOP_N,
    FAISS_TOP_K_RETRIEVE, LLM_CONTEXT_CHUNKS,
    COUNTRY_SOURCES,
    RERANK_SKIP_MARGIN, RERANK_SKIP_MIN_SCORE,
)

from modules.cache      import cache
from modules.embedder   import Embedder
from modules.faiss_index import IndexManager
from modules.reranker   import Reranker, RankedHit
from modules.crawler    import Crawler
from modules.llm        import LLM


class Pipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.index    = IndexManager()
        self.reranker = Reranker()
        self.llm      = LLM()
        self._crawler  = None

    def query(self, question: str) -> Generator[dict, None, None]:
        t0 = time.time()

        # ── Step 1: query_cache ──────────────────────────────────────────────
        cache_key = "query:" + hashlib.md5(question.encode()).hexdigest()
        cached    = cache.get(cache_key)
        if cached:
            yield _log("cache", "HIT", {"key": cache_key})
            yield {"type": "cached", "data": cached["answer"]}
            yield {"type": "sources", "data": cached["sources"]}
            return
        yield _log("cache", "MISS", {"key": cache_key})

        # ── Step 2: embed query ──────────────────────────────────────────────
        yield _log("embed", "embedding query…")
        q_emb = self.embedder.embed_query(question)
        top_sparse = sorted(
            q_emb["sparse"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        yield _log("embed", "done", {"top_sparse_tokens": top_sparse})

        # ── Step 2b: lazy-crawl country if detected ─────────────────────────
        country = self._detect_country(question)
        if country:
            for event in self._ensure_country_indexed(country):
                yield event

        # ── Step 3: FAISS search ─────────────────────────────────────────────
        yield _log("faiss", f"searching top-{FAISS_TOP_K_RETRIEVE}…")
        hits = self.index.search(q_emb, top_k=FAISS_TOP_K_RETRIEVE, index="both")
        yield _log("faiss", f"retrieved {len(hits)} hits", {
            "top5": [{"id": h.section_id, "hybrid": round(h.hybrid_score, 3)} for h in hits[:5]]
        })

        # ── Step 4: rerank (skip if FAISS scores decisive) ──────────────────
        if self._should_skip_rerank(hits):
            yield _log("rerank", "SKIPPED — FAISS scores decisive")
            ranked = [
                RankedHit(hit=h, reranker_score=h.hybrid_score)
                for h in hits[:RERANK_TOP_N]
            ]
        else:
            yield _log("rerank", f"reranking {len(hits)} hits → top-{RERANK_TOP_N}…")
            ranked = self.reranker.rerank(question, hits, top_n=RERANK_TOP_N)
            yield _log("rerank", "done", {
                "top1": {
                    "id":    ranked[0].section_id if ranked else None,
                    "score": round(ranked[0].reranker_score, 3) if ranked else None,
                }
            })

        # ── Step 5: fallback web search ──────────────────────────────────────
        if self.reranker.needs_fallback(ranked):
            yield _log("fallback", "score below threshold → web search triggered")
            try:
                web_text  = self._web_fallback(question)
                web_chunk = web_text[:2000]
                from modules.faiss_index import Hit
                web_hit = Hit(
                    section_id="WEB-FALLBACK",
                    text=web_chunk,
                    page=0,
                    doc_name="web",
                    country="WEB",
                    dense_score=0.0,
                    sparse_score=0.0,
                    hybrid_score=0.0,
                )
                ranked = [RankedHit(hit=web_hit, reranker_score=0.5)] + ranked
                yield _log("fallback", "web chunk prepended to context")
            except Exception as e:
                yield _log("fallback", f"web search failed: {e}")

        # ── Step 6: stream LLM answer ────────────────────────────────────────
        context_chunks = [r.text for r in ranked[:LLM_CONTEXT_CHUNKS]]
        sources        = [r.to_dict() for r in ranked[:LLM_CONTEXT_CHUNKS]]

        yield _log("llm", "streaming answer…")
        answer_tokens = []
        for token in self.llm.stream(question, context_chunks):
            answer_tokens.append(token)
            yield {"type": "token", "data": token}

        full_answer = "".join(answer_tokens)
        yield {"type": "sources", "data": sources}

        # ── Step 7: save to query_cache ───────────────────────────────────────
        cache.set(cache_key, {"answer": full_answer, "sources": sources}, ttl=TTL_QUERY_CACHE)
        yield _log("cache", f"saved (TTL {TTL_QUERY_CACHE}s)", {"key": cache_key})
        yield _log("done", f"total {time.time()-t0:.2f}s")

    # ── Reranker skip logic ──────────────────────────────────────────────────

    @staticmethod
    def _should_skip_rerank(hits) -> bool:
        """Skip reranker when top-1 is strong AND well-separated from top-2."""
        if len(hits) < 2:
            return False
        top1 = hits[0].hybrid_score
        top2 = hits[1].hybrid_score
        return top1 > RERANK_SKIP_MIN_SCORE and (top1 - top2) > RERANK_SKIP_MARGIN

    # ── Country lazy-crawl helpers ────────────────────────────────────────────

    def _detect_country(self, question: str) -> Optional[str]:
        q = question.lower()
        if any(k in q for k in ("thailand", "thai", "ไทย", "pdpa")):
            return "TH"
        if any(k in q for k in ("vietnam", "viet", "vietnamese", "việt", "decree 13")):
            return "VN"
        if any(k in q for k in ("indonesia", "indonesian", "pasal", "pp71")):
            return "ID"
        return None

    def _ensure_country_indexed(self, country: str) -> list[dict]:
        logs = []
        if cache.exists(f"faiss:{country}") or cache.exists(f"loaded:{country}"):
            logs.append(_log("lazy", f"{country} index loaded — skipping crawl"))
            return logs

        urls = COUNTRY_SOURCES.get(country, [])
        if not urls:
            logs.append(_log("lazy", f"{country} cache MISS — no source URLs configured, skipping"))
            return logs

        logs.append(_log("lazy", f"{country} cache MISS — crawling {len(urls)} source(s)…"))

        if self._crawler is None:
            self._crawler = Crawler()

        from modules.segmenter import Segmenter
        seg = Segmenter()

        all_segs = []
        for url in urls:
            try:
                result = self._crawler.fetch(url)
                pages  = [{"page": 1, "text": result["text"]}]
                segs   = seg.segment(pages, doc_name=url, country=country)
                all_segs.extend(segs)
                logs.append(_log("lazy", f"crawled {url} → {len(segs)} segments"))
            except Exception as e:
                logs.append(_log("lazy", f"crawl failed for {url}: {e}"))

        if not all_segs:
            logs.append(_log("lazy", f"no segments extracted for {country} — index not updated"))
            return logs

        embs = self.embedder.embed([s.text for s in all_segs])
        self.index.add([s.to_dict() for s in all_segs], embs, index="main", country=country)
        logs.append(_log("lazy", f"{country} indexed — {len(all_segs)} vectors saved permanently"))
        return logs

    def _web_fallback(self, question: str) -> str:
        search_url = f"https://duckduckgo.com/html/?q={question.replace(' ', '+')}"
        if self._crawler is None:
            self._crawler = Crawler()
        result = self._crawler.fetch(search_url)
        return result["text"]


def _log(step: str, message: str, data: any = None) -> dict:
    return {"type": "log", "step": step, "message": message, "data": data}
