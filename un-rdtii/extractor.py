"""
extractor.py — PDF text extraction for RDTII Regulatory Analyzer
Extracts and chunks text from PDF documents.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber

_OCR_AVAILABLE = False
try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from PIL import Image
    _OCR_AVAILABLE = True
    _surya_recognition = None
    _surya_detection = None
except ImportError:
    pass

# Data structures
@dataclass
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


@dataclass
class ExtractionResult:
    full_text: str
    chunks: list[TextChunk]
    page_count: int
    ocr_pages: list[int] = field(default_factory=list)
    error: Optional[str] = None


# Text cleaning helpers
_WHITESPACE_RE = re.compile(r"\s{3,}")
_PAGE_NUM_RE = re.compile(r"^\s*[\-–—]?\s*\d+\s*[\-–—]?\s*$", re.MULTILINE)
_HEADER_RE = re.compile(
    r"^\s*(page \d+|confidential|draft|version \d[\d.]*|all rights reserved)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_page_text(text: str) -> str:
    """Remove page numbers, repeated headers, and excessive whitespace."""
    text = _PAGE_NUM_RE.sub("", text)
    text = _HEADER_RE.sub("", text)
    # Collapse 3+ consecutive whitespace/newlines to two newlines
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    # Collapse multiple spaces on a single line (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_page_text_with_ocr(page_image) -> str:
    """Run Surya OCR on a single page image."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        global _surya_recognition, _surya_detection
        if _surya_recognition is None:
            _surya_recognition = RecognitionPredictor()
        if _surya_detection is None:
            _surya_detection = DetectionPredictor()

        results = _surya_recognition([page_image], det_predictor=_surya_detection)
        text_lines = []
        for line in results[0].text_lines:
            if line.text.strip():
                text_lines.append(line.text)
        return "\n".join(text_lines)
    except Exception:
        return ""

# Chunking
def _chunk_text(text: str, page_number: int, chunk_size: int = 500, overlap: int = 50) -> list[TextChunk]:
    """
    Split text into chunks of at most chunk_size characters with overlap.
    Tries to break at sentence boundaries ('. ', '? ', '! ') when possible.
    """
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Try to find a natural break point (sentence boundary) near end
            search_start = max(start, end - 80)
            best_break = -1
            for sep in (". ", "? ", "! ", "\n\n", "\n"):
                pos = text.rfind(sep, search_start, end)
                if pos != -1 and pos > best_break:
                    best_break = pos + len(sep)

            if best_break > start:
                end = best_break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        # Advance with overlap
        start = end - overlap if end < length else length

    return chunks

# Main extraction logic
def extract_text(pdf_path: str, progress_callback=None) -> ExtractionResult:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        progress_callback: Optional callable(float) receiving 0.0–1.0 progress

    Returns:
        ExtractionResult with full_text, chunks, page_count, and ocr_pages
    """
    path = Path(pdf_path)
    if not path.exists():
        return ExtractionResult(
            full_text="",
            chunks=[],
            page_count=0,
            error=f"File not found: {pdf_path}",
        )

    all_page_texts: list[str] = []
    ocr_pages: list[int] = []
    all_chunks: list[TextChunk] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)

            # Pre-convert to images if OCR is available
            page_images: list = []
            if _OCR_AVAILABLE:
                try:
                    from pdf2image import convert_from_path
                    page_images = convert_from_path(str(path), dpi=150)
                except Exception:
                    page_images = []

            for i, page in enumerate(pdf.pages):
                if progress_callback:
                    progress_callback(i / total_pages)

                raw_text = page.extract_text() or ""
                cleaned = _clean_page_text(raw_text)

                # Fall back to OCR if pdfplumber returned mostly empty/garbage
                if len(cleaned.strip()) < 30 and _OCR_AVAILABLE and i < len(page_images):
                    ocr_text = _extract_page_text_with_ocr(page_images[i])
                    cleaned = _clean_page_text(ocr_text)
                    if cleaned:
                        ocr_pages.append(i + 1)

                if cleaned:
                    all_page_texts.append(cleaned)
                    page_chunks = _chunk_text(cleaned, page_number=i + 1)
                    all_chunks.extend(page_chunks)

        if progress_callback:
            progress_callback(1.0)

        full_text = "\n\n".join(all_page_texts)
        return ExtractionResult(
            full_text=full_text,
            chunks=all_chunks,
            page_count=total_pages,
            ocr_pages=ocr_pages,
        )

    except Exception as exc:
        return ExtractionResult(
            full_text="",
            chunks=[],
            page_count=0,
            error=f"Extraction failed: {exc}",
        )


def get_chunk_dicts(result: ExtractionResult) -> list[dict]:
    """Convert chunks to plain dictionaries suitable for JSON serialization."""
    return [
        {
            "text": c.text,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in result.chunks
    ]


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        sys.exit(1)

    result = extract_text(sys.argv[1], progress_callback=lambda p: print(f"  {p*100:.0f}%", end="\r"))
    print(f"\nPages: {result.page_count}")
    print(f"OCR pages: {result.ocr_pages}")
    print(f"Chunks: {len(result.chunks)}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"\nFirst chunk:\n{result.chunks[0].text[:300] if result.chunks else '(none)'}")