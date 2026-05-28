"""
config.py — All tuneable constants for the prototype.
Edit here; never hardcode values in modules.
"""

import os
import torch

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Model IDs (HuggingFace) ──────────────────────────────────────────────────
EMBED_MODEL_ID    = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_ID      = "unsloth/Phi-3.5-mini-instruct"  # lightweight test; swap back to unsloth/Qwen2.5-7B-Instruct-bnb-4bit

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_DIM         = 1024          # BGE-M3 dense output dimension
EMBED_BATCH_SIZE  = 8             # lower on CPU, raise on GPU
SPARSE_TOP_K      = 20            # keep top-N sparse tokens per text

# ── FAISS ────────────────────────────────────────────────────────────────────
FAISS_TOP_K_RETRIEVE = 10         # initial retrieval before rerank
FAISS_NLIST          = 50         # IVF clusters (used when index > 1000 vecs)

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANK_TOP_N    = 5               # how many to keep after reranking
RERANK_THRESHOLD = 0.60           # below this → trigger web fallback

# Skip reranker when FAISS results are decisive:
#   top1 - top2 > RERANK_SKIP_MARGIN  AND  top1 > RERANK_SKIP_MIN_SCORE
RERANK_SKIP_MARGIN    = 0.15
RERANK_SKIP_MIN_SCORE = 0.70

# ── Mock Redis TTL (seconds) ──────────────────────────────────────────────────
TTL_QUERY_CACHE   = 3600          # 1 hour  — query answers
TTL_TEMP_INDEX    = 72 * 3600     # 72 hours — uploaded PDF index
TTL_COUNTRY_INDEX = None          # permanent — crawled country law

# ── Segmenter ────────────────────────────────────────────────────────────────
# Add more patterns here as needed
SEGMENT_PATTERNS = [
    r"(?i)^(section|article|มาตรา|pasal|điều|clause)\s*[\d]+",
    r"(?i)^(\d+[\.\)])\s+\w",     # numbered list like "1. " or "1) "
]
MIN_SEGMENT_CHARS = 80            # discard segments shorter than this

# ── Crawler ──────────────────────────────────────────────────────────────────
CRAWL_TIMEOUT_MS    = 30_000      # Playwright navigation timeout
CRAWL_USER_AGENT    = (
    "Mozilla/5.0 (compatible; PrototypeBot/1.0)"
)

# ── LLM Generation ───────────────────────────────────────────────────────────
LLM_MAX_NEW_TOKENS  = 512
LLM_TEMPERATURE     = 0.2
LLM_CONTEXT_CHUNKS  = 5          # how many reranked chunks to pass to LLM

# ── Country Sources (lazy crawl on first miss) ────────────────────────────────
# Add real URLs when available. Empty list = skip auto-crawl for that country.
COUNTRY_SOURCES = {
    "TH": [
        "https://ratchakitcha.soc.go.th/documents/17082307.pdf",
        "https://www.bot.or.th/content/dam/bot/fipcs/documents/FPG/2560/EngPDF/25600035.pdf",
        "https://broadcast.nbtc.go.th/data/document/law/doc/th/580300000001.pdf",
    ],
    "VN": [
        "https://eurochamvn.org/wp-content/uploads/2023/02/Decree-13-2023-PDPD_EN_clean.pdf",
    ],
    "ID": [
        "https://wplibrary.co.id/sites/default/files/PP%2071_2019%20%5BEng%5D%5BHO%5D.PDF",
    ],
}

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.expanduser("~/.prototype_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
