"""
main.py — CPU-optimized REPL entry point.

Changes from GPU version:
  - Background preload of embedder + reranker at startup
  - No GPU device switching
  - Uses ONNX models (fastembed) + llama.cpp LLM

Run:
    cd cpu && python main.py
"""

import sys
import os
import threading

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'shared'))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.prompt  import Prompt
from rich import   print as rprint

from config import DEVICE, LLM_GGUF_REPO, LLM_GGUF_PATH

console = Console()

BANNER = """[bold cyan]
          ████
         ░░███
  ██████  ░███   ██████   ████████   ██████
 ███░░███ ░███  ░░░░░███ ░░███░░███ ░░░░░███
░███ ░░░  ░███   ███████  ░███ ░░░   ███████
░███  ███ ░███  ███░░███  ░███      ███░░███
░░██████  █████░░████████ █████    ░░████████
 ░░░░░░  ░░░░░  ░░░░░░░░ ░░░░░      ░░░░░░░░
[/bold cyan]  [dim]CPU mode — ONNX embed/rerank · llama.cpp LLM[/dim]"""

HELP_TEXT = """
[bold]Available commands[/bold]

  [cyan]/upload[/cyan] [dim]<path>[/dim]          OCR → segment → embed → build Temp Index
  [cyan]/ask[/cyan] [dim]<question>[/dim]          Full RAG pipeline (cache → FAISS → rerank → LLM)
  [cyan]/search[/cyan] [dim]<query>[/dim]          FAISS search only, shows top-k hits + scores
  [cyan]/crawl[/cyan] [dim]<url> [country][/dim]   Crawl URL → segment → embed → save to Main Index
  [cyan]/rerank[/cyan] [dim]<query>[/dim]           Re-rank last /search result with BGE reranker
  [cyan]/embed[/cyan] [dim]<text>[/dim]             Show dense shape + top sparse tokens
  [cyan]/ocr[/cyan] [dim]<path>[/dim]               Run OCR only, print pages
  [cyan]/segment[/cyan] [dim]<path>[/dim]           OCR + segment, show detected sections
  [cyan]/index info[/cyan]                  Show Main + Temp index stats
  [cyan]/index clear temp[/cyan]            Drop Temp Index
  [cyan]/cache[/cyan]                       Show all cache keys + TTL
  [cyan]/cache clear[/cyan]                 Flush cache
  [cyan]/compare[/cyan]                     Compare Temp Index vs Main Index
  [cyan]/log[/cyan]                         Toggle transparency log (verbose mode)
  [cyan]/gpu[/cyan]                         Show compute device
  [cyan]/help[/cyan]                        This message
  [cyan]/exit[/cyan]                        Quit
"""


def _background_preload(repl):
    """Preload full pipeline in background thread to kill first-query latency."""
    try:
        from modules.pipeline import Pipeline
        pipe = Pipeline()
        repl._pipeline = pipe
        repl._embedder = pipe.embedder
        repl._index    = pipe.index
        repl._reranker = pipe.reranker
    except Exception:
        pass


class REPL:
    def __init__(self):
        self._verbose = False
        self._last_hits = []
        self._pipeline = None
        self._embedder = None
        self._index    = None
        self._reranker = None
        self._ocr      = None
        self._seg      = None
        self._crawler  = None

    # ── Lazy singletons ──────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            from modules.pipeline import Pipeline
            with console.status("[cyan]Loading pipeline…[/cyan]"):
                self._pipeline = Pipeline()
                self._embedder = self._pipeline.embedder
                self._index    = self._pipeline.index
                self._reranker = self._pipeline.reranker
        return self._pipeline

    def _get_embedder(self):
        if self._embedder is None:
            from modules.embedder import Embedder
            with console.status("[cyan]Loading ONNX embedder…[/cyan]"):
                self._embedder = Embedder()
        return self._embedder

    def _get_index(self):
        if self._index is None:
            from modules.faiss_index import IndexManager
            self._index = IndexManager()
        return self._index

    def _get_reranker(self):
        if self._reranker is None:
            from modules.reranker import Reranker
            with console.status("[cyan]Loading ONNX reranker…[/cyan]"):
                self._reranker = Reranker()
        return self._reranker

    def _get_ocr(self):
        if self._ocr is None:
            from modules.ocr import OCREngine
            with console.status("[cyan]Loading Surya OCR…[/cyan]"):
                self._ocr = OCREngine()
        return self._ocr

    def _get_seg(self):
        if self._seg is None:
            from modules.segmenter import Segmenter
            self._seg = Segmenter()
        return self._seg

    def _get_crawler(self):
        if self._crawler is None:
            from modules.crawler import Crawler
            self._crawler = Crawler()
        return self._crawler

    # ── Command dispatch ──────────────────────────────────────────────────────

    def dispatch(self, line: str):
        line = line.strip()
        if not line:
            return
        parts = line.split(None, 2)
        cmd   = parts[0].lower()
        args  = parts[1:] if len(parts) > 1 else []

        commands = {
            "/help":    self._cmd_help,
            "/upload":  self._cmd_upload,
            "/ask":     self._cmd_ask,
            "/search":  self._cmd_search,
            "/crawl":   self._cmd_crawl,
            "/rerank":  self._cmd_rerank,
            "/embed":   self._cmd_embed,
            "/ocr":     self._cmd_ocr,
            "/segment": self._cmd_segment,
            "/index":   self._cmd_index,
            "/cache":   self._cmd_cache,
            "/compare": self._cmd_compare,
            "/log":     self._cmd_log,
            "/gpu":     self._cmd_gpu,
            "/exit":    lambda _: sys.exit(0),
        }

        handler = commands.get(cmd)
        if handler:
            try:
                handler(args)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                if self._verbose:
                    import traceback; traceback.print_exc()
        else:
            console.print(f"[yellow]Unknown command:[/yellow] {cmd}  (type /help)")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _cmd_help(self, _):
        console.print(HELP_TEXT)

    def _cmd_gpu(self, _):
        console.print(f"[cyan]Device:[/cyan] {DEVICE}")
        console.print(f"[cyan]LLM:[/cyan] llama.cpp  {LLM_GGUF_PATH or LLM_GGUF_REPO}")

    def _cmd_log(self, _):
        self._verbose = not self._verbose
        state = "[green]ON[/green]" if self._verbose else "[dim]OFF[/dim]"
        console.print(f"[cyan]Transparency log:[/cyan] {state}")

    def _cmd_upload(self, args):
        if not args:
            console.print("[red]Usage:[/red] /upload <path> [country]"); return
        path    = args[0]
        country = args[1].upper() if len(args) > 1 else "XX"
        doc_name = os.path.basename(path)

        console.print(f"[cyan]Running OCR on {doc_name}…[/cyan]")
        pages = self._get_ocr().run(path)
        console.print(f"[green]✓[/green] OCR: {len(pages)} pages")

        console.print("[cyan]Segmenting…[/cyan]")
        segs = self._get_seg().segment(pages, doc_name=doc_name, country=country)
        console.print(f"[green]✓[/green] Segments: {len(segs)}")
        _show_segments_preview(segs[:3])

        console.print("[cyan]Embedding (ONNX)…[/cyan]")
        texts = [s.text for s in segs]
        embs  = self._get_embedder().embed(texts)

        console.print("[cyan]Building FAISS Temp Index…[/cyan]")
        self._get_index().add([s.to_dict() for s in segs], embs, index="temp")

        console.print(f"[green]✓[/green] Temp Index ready — {len(segs)} vectors")

    def _cmd_ask(self, args):
        if not args:
            console.print("[red]Usage:[/red] /ask <question>"); return
        question = " ".join(args)
        console.print(f"[dim]Query:[/dim] {question}\n")

        pipe = self._get_pipeline()
        console.print("[bold cyan]Answer[/bold cyan]")
        console.rule()

        sources = []
        for event in pipe.query(question):
            if event["type"] == "token":
                console.print(event["data"], end="", highlight=False)
            elif event["type"] == "sources":
                sources = event["data"]
            elif event["type"] == "cached":
                console.print(event["data"])
            elif event["type"] == "log" and self._verbose:
                _show_log(event)

        console.print()
        console.rule()
        _show_sources(sources)

    def _cmd_search(self, args):
        if not args:
            console.print("[red]Usage:[/red] /search <query>"); return
        query = " ".join(args)

        with console.status("[cyan]Embedding query (ONNX)…[/cyan]"):
            q_emb = self._get_embedder().embed_query(query)

        with console.status("[cyan]Searching FAISS…[/cyan]"):
            hits  = self._get_index().search(q_emb, top_k=10, index="both")

        self._last_hits = hits
        _show_hits_table(hits)

    def _cmd_rerank(self, args):
        if not self._last_hits:
            console.print("[yellow]Run /search first.[/yellow]"); return
        query  = " ".join(args) if args else Prompt.ask("Query")
        with console.status("[cyan]Reranking (ONNX)…[/cyan]"):
            ranked = self._get_reranker().rerank(query, self._last_hits)
        _show_ranked_table(ranked)

    def _cmd_embed(self, args):
        if not args:
            console.print("[red]Usage:[/red] /embed <text>"); return
        text = " ".join(args)
        with console.status("[cyan]Embedding (ONNX)…[/cyan]"):
            r = self._get_embedder().embed_query(text)
        console.print(f"[cyan]Dense:[/cyan]  shape={r['dense'].shape}  norm={float((r['dense']**2).sum()**0.5):.4f}")
        top_sparse = sorted(r["sparse"].items(), key=lambda x: x[1], reverse=True)[:10]
        t = Table("token_id", "weight", show_header=True, header_style="bold")
        for tok, w in top_sparse:
            t.add_row(tok, f"{w:.4f}")
        console.print(t)

    def _cmd_ocr(self, args):
        if not args:
            console.print("[red]Usage:[/red] /ocr <path>"); return
        with console.status("[cyan]Running OCR…[/cyan]"):
            pages = self._get_ocr().run(args[0])
        for p in pages:
            console.print(Panel(
                p["text"][:600] + ("…" if len(p["text"]) > 600 else ""),
                title=f"[cyan]Page {p['page']}[/cyan]",
                border_style="dim",
            ))

    def _cmd_segment(self, args):
        if not args:
            console.print("[red]Usage:[/red] /segment <path> [country]"); return
        path    = args[0]
        country = args[1].upper() if len(args) > 1 else "XX"
        with console.status("[cyan]OCR…[/cyan]"):
            pages = self._get_ocr().run(path)
        with console.status("[cyan]Segmenting…[/cyan]"):
            segs  = self._get_seg().segment(pages, doc_name=os.path.basename(path), country=country)
        stats = self._get_seg().stats(segs)
        console.print(f"[green]Segments:[/green] {stats['total']}  avg_chars: {stats['avg_chars']}")
        _show_segments_preview(segs[:5])

    def _cmd_crawl(self, args):
        if not args:
            console.print("[red]Usage:[/red] /crawl <url> [country]"); return
        url     = args[0]
        country = args[1].upper() if len(args) > 1 else "WEB"

        with console.status(f"[cyan]Crawling {url}…[/cyan]"):
            result = self._get_crawler().fetch(url)
        console.print(f"[green]✓[/green] type={result['type']}  chars={len(result['text'])}")

        pages = [{"page": 1, "text": result["text"]}]
        segs  = self._get_seg().segment(pages, doc_name=url, country=country)
        console.print(f"[green]✓[/green] Segments: {len(segs)}")

        if not segs:
            console.print("[yellow]No segments extracted — index not updated.[/yellow]"); return

        with console.status("[cyan]Embedding + indexing (ONNX)…[/cyan]"):
            embs = self._get_embedder().embed([s.text for s in segs])
            self._get_index().add([s.to_dict() for s in segs], embs, index="main", country=country)

        console.print(f"[green]✓[/green] Saved to Main Index [{country}]")

    def _cmd_index(self, args):
        sub = args[0] if args else "info"
        if sub == "info":
            info = self._get_index().info()
            console.print(f"[cyan]Temp Index:[/cyan] {info['temp_vectors']} vectors")
            console.print(f"[cyan]Main Index:[/cyan] {info['main_vectors']}")
            console.print(f"[cyan]Loaded countries:[/cyan] {info['loaded_countries']}")
        elif sub == "clear" and len(args) > 1:
            self._get_index().clear(args[1])
            console.print(f"[green]✓[/green] Cleared {args[1]} index")
        else:
            console.print("[red]Usage:[/red] /index info | /index clear <temp|main|all>")

    def _cmd_cache(self, args):
        from modules.cache import cache
        if args and args[0] == "clear":
            cache.flush()
            console.print("[green]✓[/green] Cache flushed")
            return
        rows = cache.info()
        if not rows:
            console.print("[dim]Cache is empty.[/dim]"); return
        t = Table("key", "ttl", "size", show_header=True, header_style="bold cyan")
        for r in rows:
            t.add_row(r["key"], r["ttl"], r["size"])
        console.print(t)

    def _cmd_compare(self, _):
        console.print("[yellow]Compare: not yet implemented — coming in next sprint.[/yellow]")
        console.print("[dim]Plan: FAISS search per indicator (10 indicators), score each country.[/dim]")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        console.print(BANNER)
        console.print(f"[dim]Device: {DEVICE}  |  LLM: {LLM_GGUF_PATH or LLM_GGUF_REPO}  |  type /help for commands[/dim]")

        # Preload full pipeline in background (models ready before first query)
        console.print("[dim]  ▸ preloading ONNX models in background…[/dim]")
        t = threading.Thread(target=_background_preload, args=(self,), daemon=True)
        t.start()

        console.print()
        while True:
            try:
                line = Prompt.ask("[bold green]>[/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                break

            if not line.startswith("/"):
                self.dispatch(f"/ask {line}")
            else:
                self.dispatch(line)


# ── Display helpers ────────────────────────────────────────────────────────────

def _show_hits_table(hits):
    if not hits:
        console.print("[dim]No results.[/dim]"); return
    t = Table("rank", "section_id", "page", "country", "hybrid", "preview",
              show_header=True, header_style="bold")
    for i, h in enumerate(hits):
        t.add_row(
            str(i+1),
            h.section_id,
            str(h.page),
            h.country,
            f"{h.hybrid_score:.3f}",
            h.text[:60] + "…",
        )
    console.print(t)

def _show_ranked_table(ranked):
    if not ranked:
        console.print("[dim]No results.[/dim]"); return
    t = Table("rank", "section_id", "page", "reranker", "preview",
              show_header=True, header_style="bold")
    for i, r in enumerate(ranked):
        t.add_row(
            str(i+1),
            r.section_id,
            str(r.page),
            f"{r.reranker_score:.3f}",
            r.text[:60] + "…",
        )
    console.print(t)

def _show_segments_preview(segs):
    for s in segs:
        console.print(Panel(
            s.text[:200] + ("…" if len(s.text) > 200 else ""),
            title=f"[cyan]{s.section_id}[/cyan]  p.{s.page}",
            border_style="dim",
        ))

def _show_sources(sources):
    if not sources:
        return
    console.print("[bold]Sources[/bold]")
    for i, s in enumerate(sources):
        score = s.get("reranker_score", s.get("hybrid_score", 0))
        console.print(
            f"  [cyan][{i+1}][/cyan] {s['section_id']}  "
            f"p.{s['page']}  {s['doc_name']}  "
            f"[dim]score={score:.3f}[/dim]"
        )

def _show_log(event):
    console.print(
        f"[dim]  ▸ [{event['step']}][/dim] {event['message']}",
        highlight=False,
    )


if __name__ == "__main__":
    REPL().run()
