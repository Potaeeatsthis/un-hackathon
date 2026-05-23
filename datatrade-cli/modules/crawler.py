"""
modules/crawler.py — Playwright web crawler for fallback search.

Triggered when FAISS + reranker confidence is below threshold.
Supports two modes:
  HTML URL  → Playwright extract visible text
  PDF URL   → download bytes → pass to ocr.py

"Crawlee" in the architecture doc refers to the JS library.
This Python equivalent uses Playwright directly, which is the
underlying engine. For a full Crawlee port, see the JS bridge note below.

JS Bridge Note (optional):
  If you want the real Crawlee JS library, you can call it via subprocess:
  `node crawlee_worker.js <url>` and parse stdout.
  For the prototype, Playwright-Python is sufficient and simpler.

Usage:
    from modules.crawler import Crawler
    c = Crawler()
    result = c.fetch("https://www.pdpa.go.th/")
    print(result["type"])   # "html" or "pdf"
    print(result["text"][:500])
"""

import io
import re
from typing import Optional
from config import CRAWL_TIMEOUT_MS, CRAWL_USER_AGENT


class Crawler:
    def __init__(self):
        self._browser = None
        self._playwright = None

    def _launch(self):
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright not installed.\n"
                "Run: pip install playwright && playwright install chromium"
            )
        self._pw      = sync_playwright().__enter__()
        self._browser = self._pw.chromium.launch(headless=True)

    def fetch(self, url: str) -> dict:
        """
        Parameters
        ----------
        url : any HTTP/HTTPS URL

        Returns
        -------
        {
          "text":   str,          # extracted plain text
          "source": str,          # original URL
          "type":   "html"|"pdf",
        }
        """
        if url.lower().endswith(".pdf") or "pdf" in url.lower():
            return self._fetch_pdf(url)
        return self._fetch_html(url)

    def _fetch_html(self, url: str) -> dict:
        self._launch()
        page = self._browser.new_page(user_agent=CRAWL_USER_AGENT)
        try:
            page.goto(url, timeout=CRAWL_TIMEOUT_MS, wait_until="domcontentloaded")
            # Remove nav/footer/script noise
            page.evaluate("""
                ['nav','footer','script','style','header','aside'].forEach(tag =>
                    document.querySelectorAll(tag).forEach(el => el.remove())
                )
            """)
            text = page.inner_text("body")
            text = self._clean(text)
        finally:
            page.close()
        return {"text": text, "source": url, "type": "html"}

    def _fetch_pdf(self, url: str) -> dict:
        """Download PDF bytes → OCR via ocr.py."""
        self._launch()
        import tempfile, os

        # Download via Playwright (respects JS redirects)
        page = self._browser.new_page(user_agent=CRAWL_USER_AGENT)
        try:
            with page.expect_download() as dl_info:
                page.goto(url, timeout=CRAWL_TIMEOUT_MS)
            download = dl_info.value
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            download.save_as(tmp_path)
        except Exception:
            # Fallback: raw HTTP download
            page.close()
            tmp_path = self._http_download(url)
        else:
            page.close()

        try:
            from modules.ocr import get_engine
            engine = get_engine()
            pages  = engine.run(tmp_path)
            text   = "\n\n".join(p["text"] for p in pages)
        finally:
            os.unlink(tmp_path)

        return {"text": text, "source": url, "type": "pdf"}

    @staticmethod
    def _http_download(url: str) -> str:
        """Fallback HTTP download without Playwright (no JS)."""
        import urllib.request, tempfile, os
        req = urllib.request.Request(url, headers={"User-Agent": CRAWL_USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name

    @staticmethod
    def _clean(text: str) -> str:
        """Strip excessive whitespace while preserving paragraph breaks."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def close(self):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.__exit__(None, None, None)
        self._browser = None
        self._pw = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
