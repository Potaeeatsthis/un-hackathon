"""
config.py — CPU-optimized configuration.

Key changes from GPU version:
  - ONNX runtime via fastembed (embedder + reranker)
  - llama.cpp for LLM (llama-cpp-python, in-process GGUF — no server)
  - SQLite persistent cache (survives restarts)
  - Reranker skip logic for high-confidence FAISS hits
"""

import os

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cpu"

# ── llama.cpp LLM (GGUF via llama-cpp-python) ────────────────────────────────
# Resolution order:
#   1. LLM_GGUF_PATH set + file exists → load that local .gguf
#   2. else download LLM_GGUF_FILE from LLM_GGUF_REPO (HuggingFace Hub) and cache
# Point LLM_GGUF_PATH at a local single-file GGUF for fully offline runs.
LLM_GGUF_PATH = os.getenv("LLM_GGUF_PATH", "")
LLM_GGUF_REPO = os.getenv("LLM_GGUF_REPO", "Qwen/Qwen2.5-7B-Instruct-GGUF")
LLM_GGUF_FILE = os.getenv("LLM_GGUF_FILE", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
LLM_N_CTX     = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 4)))

# ── Embedder (fastembed ONNX) ───────────────────────────────────────────────
EMBED_MODEL_ID    = "jinaai/jina-embeddings-v3"
EMBED_DIM         = 1024
EMBED_BATCH_SIZE  = 8
SPARSE_TOP_K      = 20

# ── FAISS ────────────────────────────────────────────────────────────────────
FAISS_TOP_K_RETRIEVE = 10
FAISS_NLIST          = 50

# ── Reranker (fastembed ONNX) ───────────────────────────────────────────────
RERANKER_MODEL_ID  = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_TOP_N       = 5
RERANK_THRESHOLD   = 0.60

# Skip reranker when FAISS results are decisive:
#   top1 - top2 > RERANK_SKIP_MARGIN  AND  top1 > RERANK_SKIP_MIN_SCORE
RERANK_SKIP_MARGIN    = 0.15
RERANK_SKIP_MIN_SCORE = 0.70

# ── Cache (SQLite) ──────────────────────────────────────────────────────────
TTL_QUERY_CACHE   = 3600
TTL_TEMP_INDEX    = 72 * 3600
TTL_COUNTRY_INDEX = None

# ── Segmenter ────────────────────────────────────────────────────────────────
SEGMENT_PATTERNS = [
    r"(?i)^(section|article|มาตรา|pasal|điều|clause)\s*[\d]+",
    r"(?i)^(\d+[\.\)])\s+\w",
]
MIN_SEGMENT_CHARS = 80

# ── Crawler ──────────────────────────────────────────────────────────────────
CRAWL_TIMEOUT_MS = 30_000
CRAWL_USER_AGENT = "Mozilla/5.0 (compatible; PrototypeBot/1.0)"

# ── LLM Generation ──────────────────────────────────────────────────────────
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE    = 0.2
LLM_CONTEXT_CHUNKS = 5

# ── Country Sources ──────────────────────────────────────────────────────────
COUNTRY_SOURCES = {
    "TH": [],
    "VN": [],
    "ID": [],
}

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.expanduser("~/.prototype_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_DB = os.path.join(CACHE_DIR, "cache.db")
