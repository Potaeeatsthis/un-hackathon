"""
preload.py — One-time setup script. Run before demo day.

Crawls all configured country source URLs, segments, embeds, and saves
FAISS indexes to ~/.prototype_cache/faiss_{country}.pkl so the system
boots instantly without crawling or OCR at runtime.

Usage:
    python preload.py           # process all countries
    python preload.py TH        # process one country only
"""

import os
import sys
import pickle
import time

sys.path.insert(0, os.path.dirname(__file__))

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
        return

    print("Loading models (this happens once)…")
    ocr = OCREngine()
    seg = Segmenter()
    emb = Embedder()

    t0 = time.time()
    total = 0
    for country, urls in countries.items():
        total += preload_country(country, urls, ocr, seg, emb)

    elapsed = time.time() - t0
    print(f"\nDone. {total} total vectors in {elapsed:.0f}s")
    print(f"Index files saved to: {CACHE_DIR}")


if __name__ == "__main__":
    main()
