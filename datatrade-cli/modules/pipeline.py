"""
modules/pipeline.py — Orchestrates the full RAG query flow.

Step-by-step:
  1.  Hash query → check query_cache (mock Redis)
  2.  Embed query (BGE-M3 dense + sparse)
  3.  FAISS search (Main + Temp indexes)
  4.  BGE reranker → top-N chunks
  5.  Fallback web search if confidence < threshold
  6.  Stream answer via Qwen2.5
  7.  Save result to query_cache

Each step emits a LogEntry so the REPL can show a Transparency tab.

Usage:
    from modules.pipeline import Pipeline
    pipe = Pipeline()

    for event in pipe.query("ต้องแต่งตั้ง DPO ไหม"):
        if event["type"] == "token":
            print(event["data"], end="", flush=True)
        elif event["type"] == "sources":
            print("\\nSources:", event["data"])
        elif event["type"] == "log":
            ...   # transparency info
"""

import hashlib
import time
from typing import Generator
from config import (
    TTL_QUERY_CACHE, RERANK_TOP_N,
    FAISS_TOP_K_RETRIEVE, LLM_CONTEXT_CHUNKS,
)

from modules.cache      import cache
from modules.embedder   import Embedder
from modules.faiss_index import IndexManager
from modules.reranker   import Reranker
from modules.crawler    import Crawler
from modules.llm        import LLM


class Pipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.index    = IndexManager()
        self.reranker = Reranker()
        self.llm      = LLM()
        self._crawler  = None   # lazy — only created if fallback triggered

    def query(self, question: str) -> Generator[dict, None, None]:
        """
        Yields event dicts:
          {"type": "log",     "step": str, "data": any}
          {"type": "token",   "data": str}
          {"type": "sources", "data": list[dict]}
          {"type": "cached",  "data": str}
          {"type": "error",   "data": str}
        """
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

        # ── Step 3: FAISS search ─────────────────────────────────────────────
        yield _log("faiss", f"searching top-{FAISS_TOP_K_RETRIEVE}…")
        hits = self.index.search(q_emb, top_k=FAISS_TOP_K_RETRIEVE, index="both")
        yield _log("faiss", f"retrieved {len(hits)} hits", {
            "top5": [{"id": h.section_id, "hybrid": round(h.hybrid_score, 3)} for h in hits[:5]]
        })

        # ── Step 4: rerank ───────────────────────────────────────────────────
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
                # Convert to a pseudo-Hit for uniform handling
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
                from modules.reranker import RankedHit
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

    # ── Fallback helpers ──────────────────────────────────────────────────────

    def _web_fallback(self, question: str) -> str:
        """
        Simple DuckDuckGo search → crawl top result.
        In production swap for a proper search API (SerpAPI, Brave, etc.)
        """
        search_url = f"https://duckduckgo.com/html/?q={question.replace(' ', '+')}"
        if self._crawler is None:
            self._crawler = Crawler()
        result = self._crawler.fetch(search_url)
        return result["text"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(step: str, message: str, data: any = None) -> dict:
    return {"type": "log", "step": step, "message": message, "data": data}
