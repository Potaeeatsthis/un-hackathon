"""
preload.py — One-time setup script for CPU mode.

Crawls all configured country source URLs, segments, embeds (ONNX),
and saves FAISS indexes. Also downloads the llama.cpp GGUF model.

Usage:
    cd cpu && python preload.py           # all countries
    cd cpu && python preload.py TH        # one country
"""

import os
import sys
import pickle
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'shared'))

from config import COUNTRY_SOURCES, CACHE_DIR
from modules.ocr import OCREngine
from modules.segmenter import Segmenter
from modules.embedder import Embedder
from modules.faiss_index import _SingleIndex, EMBED_DIM


def preload_country(country: str, urls: list[str], ocr, seg, emb) -> int:
    print(f"\n[{country}] Starting — {len(urls)} source(s)")

    index = _SingleIndex(EMBED_DIM)
    total_segs = 0

    for url in urls:
        print(f"  Fetching: {url}")
        try:
            from modules.crawler import Crawler
            result = Crawler().fetch(url)
            pages = [{"page": 1, "text": result["text"]}]
        except Exception as e:
            print(f"  ✗ Crawl failed: {e}")
            continue

        segs = seg.segment(pages, doc_name=url, country=country)
        if not segs:
            print(f"  ✗ No segments extracted")
            continue
        print(f"  ✓ {len(segs)} segments")

        texts = [s.text for s in segs]
        embs = emb.embed(texts)
        index.add([s.to_dict() for s in segs], embs["dense"], embs["sparse"])
        total_segs += len(segs)

    if total_segs == 0:
        print(f"  ✗ [{country}] Nothing indexed — skipping save")
        return 0

    out_path = os.path.join(CACHE_DIR, f"faiss_{country}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(index.to_bytes(), f)
    print(f"  ✓ [{country}] Saved {total_segs} vectors → {out_path}")
    return total_segs


def main():
    target = sys.argv[1].upper() if len(sys.argv) > 1 else None

    countries = {
        k: v for k, v in COUNTRY_SOURCES.items()
        if (target is None or k == target) and v
    }

    if not countries:
        if target:
            print(f"No URLs configured for {target} in COUNTRY_SOURCES.")
        else:
            print("No URLs configured in COUNTRY_SOURCES. Add URLs to config.py first.")

    # ── Load ONNX models ────────────────────────────────────────────────────
    print("Loading ONNX models (first time downloads ~1-2 GB)…")
    ocr = OCREngine()
    seg = Segmenter()

    print("  Loading ONNX embedder (BAAI/bge-m3)…")
    emb = Embedder()
    emb.embed(["warmup"])

    print("  Loading ONNX reranker (BAAI/bge-reranker-v2-m3)…")
    from modules.reranker import Reranker
    rrk = Reranker()
    rrk._load()

    # ── Load llama.cpp GGUF model ─────────────────────────────────────────────
    print("  Loading llama.cpp LLM (first time downloads the GGUF)…")
    from modules.llm import LLM
    try:
        LLM()._load()
        print(f"  ✓ llama.cpp model ready")
    except ImportError as e:
        print(f"  ⚠ {e}")

    print("All models ready.\n")

    # ── Index countries ─────────────────────────────────────────────────────
    if countries:
        t0 = time.time()
        total = 0
        for country, urls in countries.items():
            total += preload_country(country, urls, ocr, seg, emb)

        elapsed = time.time() - t0
        print(f"\nDone. {total} total vectors in {elapsed:.0f}s")
        print(f"Index files saved to: {CACHE_DIR}")
    else:
        print("No countries to index. Preload complete (models only).")


if __name__ == "__main__":
    main()
