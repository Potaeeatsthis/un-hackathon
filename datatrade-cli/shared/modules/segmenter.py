"""
modules/segmenter.py — Pattern-based legal document segmenter.

Detects section boundaries using regex (Section / Article / มาตรา / Pasal / Điều).
"""

import re
from dataclasses import dataclass
from typing import Optional
from config import SEGMENT_PATTERNS, MIN_SEGMENT_CHARS

try:
    from segmenter_rs import segment as _rust_segment
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


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
    _PATTERNS = [re.compile(p, re.MULTILINE) for p in SEGMENT_PATTERNS]

    def segment(
        self,
        pages: list[dict],
        doc_name: str,
        country: str = "XX",
    ) -> list[Segment]:
        if _USE_RUST:
            raw = _rust_segment(pages, doc_name, country, MIN_SEGMENT_CHARS)
            return [Segment(**r) for r in raw]

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
        line = line.strip()
        for pat in self._PATTERNS:
            m = pat.match(line)
            if m:
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
