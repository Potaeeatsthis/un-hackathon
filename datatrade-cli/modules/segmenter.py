"""
modules/segmenter.py — Pattern-based legal document segmenter.

Detects section boundaries using regex (Section / Article / มาตรา / Pasal / Điều).
Produces chunks with metadata that FAISS and the audit log can reference.

Output schema per segment:
    {
      "section_id": "TH-มาตรา-28",
      "page":        5,
      "text":        "มาตรา 28 ผู้ควบคุมข้อมูลส่วนบุคคล...",
      "doc_name":    "PDPA_2562.pdf",
      "country":     "TH",
    }

Usage:
    from modules.segmenter import Segmenter
    seg = Segmenter()
    segments = seg.segment(pages, doc_name="PDPA.pdf", country="TH")
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from config import SEGMENT_PATTERNS, MIN_SEGMENT_CHARS


@dataclass
class Segment:
    section_id: str
    page: int
    text: str
    doc_name: str
    country: str

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "page": self.page,
            "text": self.text,
            "doc_name": self.doc_name,
            "country": self.country,
        }


class Segmenter:
    # Compiled once at class level
    _PATTERNS = [re.compile(p, re.MULTILINE) for p in SEGMENT_PATTERNS]

    def segment(
        self,
        pages: list[dict],          # output of ocr.run()
        doc_name: str,
        country: str = "XX",
    ) -> list[Segment]:
        """
        Split OCR output into legal sections.

        Strategy:
        1. Join all pages into one stream, keeping page boundaries tagged.
        2. Scan line-by-line for a pattern match → start a new segment.
        3. Anything before the first match → prepended to first segment.
        """
        # Build a flat line list: (line_text, page_number)
        lines: list[tuple[str, int]] = []
        for p in pages:
            for line in p["text"].splitlines():
                lines.append((line, p["page"]))

        segments: list[Segment] = []
        current_lines: list[str] = []
        current_page: int = pages[0]["page"] if pages else 1
        current_id: str = f"{country}-PREAMBLE"
        seq: int = 0

        for line_text, page_num in lines:
            hit = self._match_header(line_text)
            if hit:
                # Flush previous segment
                chunk = "\n".join(current_lines).strip()
                if len(chunk) >= MIN_SEGMENT_CHARS:
                    segments.append(Segment(
                        section_id=current_id,
                        page=current_page,
                        text=chunk,
                        doc_name=doc_name,
                        country=country,
                    ))
                seq += 1
                current_id   = f"{country}-{hit}-{seq}"
                current_lines = [line_text]
                current_page  = page_num
            else:
                current_lines.append(line_text)

        # Flush last segment
        chunk = "\n".join(current_lines).strip()
        if len(chunk) >= MIN_SEGMENT_CHARS:
            segments.append(Segment(
                section_id=current_id,
                page=current_page,
                text=chunk,
                doc_name=doc_name,
                country=country,
            ))

        return segments

    def _match_header(self, line: str) -> Optional[str]:
        """
        Returns a normalised header label if line matches a section pattern,
        e.g. "มาตรา-28" or "Article-7".
        """
        line = line.strip()
        for pat in self._PATTERNS:
            m = pat.match(line)
            if m:
                # Grab first 40 chars, sanitise for use as an ID token
                label = re.sub(r"\s+", "-", line[:40]).strip("-")
                return label
        return None

    def stats(self, segments: list[Segment]) -> dict:
        return {
            "total": len(segments),
            "pages_covered": sorted({s.page for s in segments}),
            "countries": list({s.country for s in segments}),
            "avg_chars": (
                sum(len(s.text) for s in segments) // len(segments)
                if segments else 0
            ),
        }
