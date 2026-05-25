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

# Section-based chunking
# Matches Thai มาตรา, English Article/Section/Chapter headings
_SECTION_RE = re.compile(
    r"(?:^|\n)"
    r"(?:มาตรา\s+[๐-๙\d]+"
    r"|Article\s+\d+\.?"
    r"|Section\s+\d+\.?"
    r"|Chapter\s+[IVXivx\d]+"
    r")",
    re.MULTILINE | re.IGNORECASE,
)


def _chunk_by_sections(
    page_texts: list[tuple[int, str]],
) -> list[TextChunk]:
    """
    Split text at legal section/article headings (มาตรา, Article, Section, Chapter).
    Each section becomes one chunk attributed to the page where it starts.
    Falls back to paragraph splitting if no section headers are found.
    """
    import bisect

    # Combine all pages into a single stream, tracking page boundaries
    combined = ""
    offsets: list[int] = []   # char offset where each page starts
    page_nums: list[int] = []
    for page_num, text in page_texts:
        offsets.append(len(combined))
        page_nums.append(page_num)
        combined += text + "\n\n"

    def _page_at(offset: int) -> int:
        idx = bisect.bisect_right(offsets, offset) - 1
        return page_nums[max(idx, 0)]

    matches = list(_SECTION_RE.finditer(combined))

    # No section headers found — fall back to paragraph splitting
    if not matches:
        chunks: list[TextChunk] = []
        idx = 0
        for page_num, text in page_texts:
            for para in re.split(r"\n{2,}", text):
                para = para.strip()
                if para:
                    chunks.append(TextChunk(text=para, page_number=page_num, chunk_index=idx))
                    idx += 1
        return chunks

    # Build section boundaries: list of (start, end) in combined text
    boundaries = [(m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(combined))
                  for i, m in enumerate(matches)]

    # Also capture any preamble text before the first section header
    chunks = []
    idx = 0
    preamble = combined[: matches[0].start()].strip()
    if preamble:
        chunks.append(TextChunk(text=preamble, page_number=page_nums[0], chunk_index=idx))
        idx += 1

    for start, end in boundaries:
        section_text = combined[start:end].strip()
        if section_text:
            chunks.append(TextChunk(text=section_text, page_number=_page_at(start), chunk_index=idx))
            idx += 1

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

    page_texts: list[tuple[int, str]] = []  # (page_number, cleaned_text)
    ocr_pages: list[int] = []

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
                    page_texts.append((i + 1, cleaned))

        if progress_callback:
            progress_callback(1.0)

        all_chunks = _chunk_by_sections(page_texts)
        full_text = "\n\n".join(text for _, text in page_texts)
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