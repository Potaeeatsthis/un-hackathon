# LLM Chatbot — CLI Prototype

> Claude Code–style REPL for testing OCR, embedding, FAISS, crawl, and RAG pipeline logic.

---

## Directory Layout

```
prototype/
├── PROTOTYPE.md          ← this file
├── main.py               ← REPL entry point  (python main.py)
├── config.py             ← all tuneable constants
├── requirements.txt
└── modules/
    ├── cache.py          ← mock Redis (in-memory dict + TTL)
    ├── ocr.py            ← Surya PDF/image → text + page_number
    ├── segmenter.py      ← regex Section/Article/มาตรา/Pasal detection
    ├── embedder.py       ← BGE-M3  dense + sparse
    ├── faiss_index.py    ← main index + temp index management
    ├── reranker.py       ← BGE-reranker-v2-m3
    ├── crawler.py        ← Playwright fallback web search
    ├── llm.py            ← Qwen2.5-7B-Instruct (streaming)
    └── pipeline.py       ← orchestrates full RAG query flow
```

---

## Slash Commands (REPL)

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/upload <path>` | OCR a PDF/image → segment → embed → build Temp Index |
| `/ask <question>` | Full RAG pipeline (cache → embed → FAISS → rerank → LLM) |
| `/search <query>` | FAISS search only (skip LLM), shows top-k chunks + scores |
| `/crawl <url>` | Crawl a URL, segment, embed, save to Main Index as country cache |
| `/rerank <query>` | Re-rank last `/search` result with BGE reranker |
| `/cache` | Show all mock-Redis keys + TTL status |
| `/cache clear` | Flush mock-Redis |
| `/index info` | Show Main Index and Temp Index stats |
| `/index clear temp` | Drop Temp Index |
| `/ocr <path>` | Run OCR only, pretty-print extracted pages |
| `/segment <path>` | OCR + segment only, show detected sections |
| `/embed <text>` | Embed a single string, show dense shape + top sparse tokens |
| `/compare` | Compare Temp Index vs Main Index per indicator |
| `/log` | Toggle verbose execution log (transparency mode) |
| `/gpu` | Show device being used (CUDA / MPS / CPU) |
| `/exit` | Quit |

---

## Data Flow

```
/upload my_law.pdf
  └─ ocr.py          → [{text, page}]
  └─ segmenter.py    → [{section_id, text, page, doc_name}]
  └─ embedder.py     → [{dense, sparse}]
  └─ faiss_index.py  → Temp Index (in RAM, saved to cache TTL 72h)

/ask "ต้องแต่งตั้ง DPO ไหม"
  └─ embedder.py     → query_vec (dense + sparse)
  └─ cache.py        → query_cache HIT? → return instantly
  └─ faiss_index.py  → top-20 chunks (Main + Temp)
  └─ reranker.py     → top-5 reranked
  └─ crawler.py      → fallback if top score < threshold
  └─ llm.py          → streamed answer
  └─ cache.py        → save result (TTL 1h)

/crawl https://example.go.th/pdpa.html
  └─ crawler.py      → raw text
  └─ segmenter.py    → sections
  └─ embedder.py     → vectors
  └─ faiss_index.py  → save to Main Index [faiss:TH]
  └─ cache.py        → set loaded:TH = True (permanent)
```

---

## Module Contracts (Quick Reference)

### cache.py
```python
cache.get(key)              → value | None
cache.set(key, value, ttl)  → None          # ttl=None = permanent
cache.exists(key)           → bool
cache.keys(pattern)         → list[str]
cache.flush()               → None
```

### ocr.py
```python
ocr.run(path: str)  → list[{"page": int, "text": str}]
```

### segmenter.py
```python
seg.segment(pages, doc_name, country)
  → list[{"section_id": str, "page": int, "text": str, "doc_name": str, "country": str}]
```

### embedder.py
```python
emb.embed(texts: list[str])
  → {"dense": np.ndarray [N,1024], "sparse": list[dict[str,float]]}

emb.embed_query(text: str)
  → {"dense": np.ndarray [1,1024], "sparse": dict[str,float]}
```

### faiss_index.py
```python
idx.add(segments, embeddings, index="temp"|"main")  → None
idx.search(query_emb, top_k, index="both")          → list[Hit]
idx.info()                                           → dict
idx.clear(index="temp")                             → None
# Hit = {"section_id", "text", "score", "page", "doc_name", "country"}
```

### reranker.py
```python
rrk.rerank(query: str, hits: list[Hit], top_n: int) → list[Hit]
```

### crawler.py
```python
crawl.fetch(url: str)  → {"text": str, "source": str, "type": "html"|"pdf"}
```

### llm.py
```python
# sync generator — yields token strings
for token in llm.stream(prompt, context_chunks):
    print(token, end="", flush=True)
```

### pipeline.py
```python
pipeline.query(question: str)  → generator[str]   # streamed answer + sources
```

---

## Environment Setup

```bash
# Python 3.11+
pip install -r requirements.txt

# Required after pip install — Playwright needs its browser binaries
playwright install chromium

# For GPU (optional — CPU works without this)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Optional: Rust segmenter (faster, same output)

`segmenter.py` automatically uses a Rust-compiled extension if available, otherwise falls back to pure Python — no action required to run the app.

To enable the Rust path (~3× faster segmentation):

```bash
# 1. Install Rust (skip if already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install maturin (Rust→Python build tool)
pip install maturin

# 3. Build and install the extension
maturin build --release -m datatrade-cli/segmenter_rs/Cargo.toml
pip install datatrade-cli/segmenter_rs/target/wheels/*.whl
```

After this, `from segmenter_rs import segment` succeeds and `_USE_RUST = True` activates automatically.

### requirements.txt covers
- `surya-ocr`           — OCR
- `FlagEmbedding`       — BGE-M3 dense + sparse
- `faiss-cpu` / `faiss-gpu` — vector search
- `transformers`        — Qwen2.5 + BGE reranker
- `playwright`          — web crawl fallback
- `rich`                — REPL rendering (panels, spinners, syntax)
- `typer`               — CLI bootstrap
- `numpy`, `torch`

---

## Testing Each Module in Isolation

```bash
# Test OCR only
python -c "from modules.ocr import OCREngine; o=OCREngine(); print(o.run('sample.pdf'))"

# Test embedder
python -c "
from modules.embedder import Embedder
e = Embedder()
r = e.embed_query('rules of origin for export')
print('dense shape:', r['dense'].shape)
print('top sparse:', sorted(r['sparse'].items(), key=lambda x:-x[1])[:5])
"

# Test FAISS round-trip
python -c "
from modules.embedder import Embedder
from modules.faiss_index import IndexManager
e = Embedder(); idx = IndexManager()
segs = [{'section_id':'s1','text':'Data controller must appoint DPO','page':1,'doc_name':'test','country':'TH'}]
embs = e.embed([s['text'] for s in segs])
idx.add(segs, embs, index='temp')
q = e.embed_query('who appoints DPO')
hits = idx.search(q, top_k=3)
for h in hits: print(h)
"

# Full REPL
python main.py
```
