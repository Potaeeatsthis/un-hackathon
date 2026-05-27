"""
modules/ocr.py — Surya OCR wrapper.
Input : PDF path or image path (jpg/png/webp).
Output: list of {"page": int, "text": str}
"""

import os
from pathlib import Path
from typing import Optional
from config import DEVICE


class OCREngine:
    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from surya.recognition import FoundationPredictor, RecognitionPredictor
            from surya.detection import DetectionPredictor
            self._det   = DetectionPredictor()
            self._model = RecognitionPredictor(FoundationPredictor())
        except ImportError:
            raise ImportError(
                "surya-ocr not installed.\n"
                "Run: pip install surya-ocr"
            )

    def run(self, path: str, langs: Optional[list[str]] = None) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            fast = self._run_pymupdf(path)
            if fast:
                return fast
            self._load()
            return self._run_pdf(path, langs)
        elif ext in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
            self._load()
            return self._run_image(path, langs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _run_pymupdf(self, path: Path) -> list[dict]:
        try:
            import fitz
        except ImportError:
            return []

        pages = []
        with fitz.open(str(path)) as doc:
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                pages.append({"page": i + 1, "text": text})

        avg_chars = sum(len(p["text"]) for p in pages) / max(len(pages), 1)
        if avg_chars < 100:
            return []
        return pages

    def _run_pdf(self, path: Path, langs) -> list[dict]:
        try:
            from surya.input.load import load_from_file
        except ImportError:
            from pdf2image import convert_from_path
            images = convert_from_path(str(path))
            return self._run_images(images, langs)

        images, _ = load_from_file(str(path))
        return self._run_images(images, langs)

    def _run_image(self, path: Path, langs) -> list[dict]:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self._run_images([img], langs)

    def _run_images(self, images, langs) -> list[dict]:
        predictions = self._model(images, det_predictor=self._det)
        pages = []
        for i, pred in enumerate(predictions):
            text = "\n".join(
                line.text for line in pred.text_lines if line.text.strip()
            )
            pages.append({"page": i + 1, "text": text})
        return pages


_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine
