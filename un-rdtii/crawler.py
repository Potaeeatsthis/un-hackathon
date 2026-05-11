r"""
crawler.py — PDF downloader for RDTII Regulatory Analyzer
Downloads regulatory documents from SOURCES into data/documents/
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse, urljoin

DOCUMENTS_DIR = Path(__file__).parent / "data" / "documents"

SOURCES = {
    "Thailand": {
        "name": "Personal Data Protection Act B.E. 2562",
        "url": "https://ratchakitcha.soc.go.th/documents/17082307.pdf",
        "language": "en",
    },
    "Vietnam": {
        "name": "Decree 13 on Personal Data Protection",
        "url": "https://eurochamvn.org/wp-content/uploads/2023/02/Decree-13-2023-PDPD_EN_clean.pdf",
        "language": "en",
    },
    "Indonesia": {
        "name": "Government Regulation PP 71 2019",
        "url": "https://wplibrary.co.id/sites/default/files/PP 71_2019 [Eng][HO].PDF",
        "language": "en",
    },
}

# Constants
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

PDF_KEYWORDS = ["personal data", "data protection", "cross-border", "privacy", "cybersecurity"]


# Helper functions

def _ensure_dir() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _dest_path(country: str) -> Path:
    return DOCUMENTS_DIR / f"{country.lower()}.pdf"


def _try_direct_download(url: str, dest: Path, progress_callback=None) -> bool:
    """Attempt to download the PDF directly from the given URL."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type and "pdf" not in content_type:
            return False

        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunks = []

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                if progress_callback and total:
                    progress_callback(downloaded / total)

        data = b"".join(chunks)

        # Verify it looks like a PDF
        if not data.startswith(b"%PDF"):
            return False

        dest.write_bytes(data)
        return True

    except Exception:
        return False


def _search_domain_for_pdf(base_url: str, progress_callback=None) -> str | None:
    """
    Crawl the base domain looking for PDF links that match known keywords.
    Returns the first matching PDF URL or None.
    """
    parsed = urlparse(base_url)
    domain_root = f"{parsed.scheme}://{parsed.netloc}"

    try:
        resp = requests.get(domain_root, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    candidates = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].lower()
        text = tag.get_text(strip=True).lower()
        combined = href + " " + text
        if href.endswith(".pdf") and any(kw in combined for kw in PDF_KEYWORDS):
            full_url = urljoin(domain_root, tag["href"])
            candidates.append(full_url)

    for candidate in candidates:
        if _try_direct_download(candidate, Path("/tmp/_rdtii_probe.pdf"), progress_callback):
            return candidate

    return None


def download_document(country: str, progress_callback=None) -> str:
    """
    Download the PDF for the given country.

    Args:
        country: One of 'Thailand', 'Vietnam', 'Indonesia'
        progress_callback: Optional callable(float) receiving 0.0–1.0 download progress

    Returns:
        Absolute path to the saved PDF file

    Raises:
        ValueError: If country is not in SOURCES
        RuntimeError: If the document cannot be downloaded
    """
    if country not in SOURCES:
        raise ValueError(f"Unknown country '{country}'. Choose from: {list(SOURCES.keys())}")

    _ensure_dir()
    source = SOURCES[country]
    dest = _dest_path(country)

    if dest.exists() and dest.stat().st_size > 1024:
        return str(dest)

    url = source["url"]

    if progress_callback:
        progress_callback(0.0)

    success = _try_direct_download(url, dest, progress_callback)

    if not success:
        fallback_url = _search_domain_for_pdf(url, progress_callback)
        if fallback_url:
            success = _try_direct_download(fallback_url, dest, progress_callback)

    if not success or not dest.exists() or dest.stat().st_size < 1024:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download '{source['name']}' for {country}.\n"
            f"Please download manually from:\n  {url}\n"
            f"and save it to:\n  {dest}"
        )

    return str(dest)


def download_all(progress_callback=None) -> dict[str, str]:
    """
    Download PDFs for all countries in SOURCES.

    Args:
        progress_callback: Optional callable(country, float) receiving country name and progress

    Returns:
        Dictionary mapping country name to local file path
    """
    results = {}
    for country in SOURCES:
        cb = (lambda c: lambda p: progress_callback(c, p))(country) if progress_callback else None
        try:
            path = download_document(country, progress_callback=cb)
            results[country] = path
        except RuntimeError as exc:
            results[country] = f"ERROR: {exc}"
    return results


def get_file_info(country: str) -> dict:
    """Return metadata about the local file for the given country."""
    dest = _dest_path(country)
    source = SOURCES.get(country, {})
    return {
        "country": country,
        "document_name": source.get("name", ""),
        "url": source.get("url", ""),
        "local_path": str(dest),
        "downloaded": dest.exists() and dest.stat().st_size > 1024,
        "file_size_kb": round(dest.stat().st_size / 1024, 1) if dest.exists() else 0,
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    countries = [target] if target else list(SOURCES.keys())

    for c in countries:
        print(f"Downloading {c}...")
        try:
            path = download_document(c, progress_callback=lambda p: print(f"  {p*100:.0f}%", end="\r"))
            print(f"  Saved: {path}")
        except RuntimeError as e:
            print(f"  FAILED: {e}")
