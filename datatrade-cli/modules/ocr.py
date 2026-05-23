"""
modules/ocr.py — Surya OCR wrapper.
Input : PDF path or image path (jpg/png/webp).
Output: list of {"page": int, "text": str}

Surya handles multi-language natively and runs on CPU or GPU.
Model weights are downloaded on first use (~1 GB).

Usage:
    from modules.ocr import OCREngine
    engine = OCREngine()
    pages = engine.run("law.pdf")
    for p in pages:
        print(p["page"], p["text"][:80])
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
        """Lazy-load Surya models on first call."""
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
        """
        Parameters
        ----------
        path  : path to PDF or image file
        langs : ISO 639-1 list e.g. ["th", "en"]. None = auto-detect.

        Returns
        -------
        list of {"page": int, "text": str}
        """
        self._load()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._run_pdf(path, langs)
        elif ext in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}:
            return self._run_image(path, langs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

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


# ── Module-level convenience ───────────────────────────────────────────────────
_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine
