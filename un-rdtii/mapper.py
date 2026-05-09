"""
mapper.py — AI mapping to RDTII indicators using Hugging Face zero-shot classification
Uses facebook/bart-large-mnli to map text chunks to indicators without any API key.
"""

import json
from pathlib import Path
from typing import Optional

from transformers import pipeline

from extractor import TextChunk

# Constants

RESULTS_DIR = Path(__file__).parent / "data" / "results"

INDICATORS = {
    "6.1": "Ban and local processing requirements — rules that ban or restrict processing data outside the country",
    "6.2": "Local storage requirements — rules requiring data to be stored physically within the country",
    "6.3": "Infrastructure requirements — rules requiring local servers or data centers",
    "6.4": "Conditional flow regimes — rules allowing data transfers only under specific conditions like consent or adequacy",
    "6.5": "Not participating in data transfer agreements — absence of treaty commitments on cross-border data flows",
    "7.1": "Lack of comprehensive data protection framework — gaps or incompleteness in the overall legal framework",
    "7.2": "Lack of dedicated cybersecurity framework — no specific cybersecurity law exists",
    "7.3": "Data retention requirements — rules about minimum or maximum periods to keep personal data",
    "7.4": "DPIA or DPO requirements — rules requiring data protection impact assessments or a data protection officer",
    "7.5": "Government access to personal data — rules allowing authorities to request or access private data",
}

MODEL_NAME = "facebook/bart-large-mnli"
CONFIDENCE_THRESHOLD = 0.5
PRIMARY_THRESHOLD = 0.85
CONTEXTUAL_THRESHOLD = 0.70
BATCH_SIZE = 8

# Module-level cache so the model is only loaded once per process
_classifier = None


# Helper functions
def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,  # CPU only; set to 0 for GPU
        )
    return _classifier

def _score_to_match_level(score: float) -> str:
    if score >= PRIMARY_THRESHOLD:
        return "Primary"
    if score >= CONTEXTUAL_THRESHOLD:
        return "Contextual"
    if score >= CONFIDENCE_THRESHOLD:
        return "Implicit"
    return "No match"

# Main functions

def map_chunks(
    chunks: list[TextChunk],
    country: str,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Run zero-shot classification on each chunk against all 10 indicators.

    Args:
        chunks: List of TextChunk objects from extractor.py
        country: Country name used for saving results
        progress_callback: Optional callable(float) receiving 0.0–1.0 progress

    Returns:
        Dictionary of indicator_id -> match result dict
    """
    classifier = _get_classifier()
    candidate_labels = list(INDICATORS.values())
    indicator_ids = list(INDICATORS.keys())

    # best_match[indicator_id] = {"score": float, "chunk": TextChunk}
    best_match: dict[str, dict] = {}

    total = len(chunks)
    processed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [c.text for c in batch]

        try:
            # Run classification for each text in the batch
            for idx, (text, chunk) in enumerate(zip(batch_texts, batch)):
                result = classifier(text, candidate_labels, multi_label=True)

                for label, score in zip(result["labels"], result["scores"]):
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    # Map label back to indicator id
                    label_idx = candidate_labels.index(label)
                    ind_id = indicator_ids[label_idx]

                    if ind_id not in best_match or score > best_match[ind_id]["score"]:
                        best_match[ind_id] = {"score": score, "chunk": chunk}

                processed += 1
                if progress_callback:
                    progress_callback(processed / total)

        except Exception as exc:
            # Skip bad batch; continue processing
            processed += len(batch)
            if progress_callback:
                progress_callback(processed / total)
            continue

    # Build final output dict
    output: dict[str, dict] = {}
    for ind_id, indicator_label in INDICATORS.items():
        if ind_id not in best_match:
            continue
        match = best_match[ind_id]
        score = match["score"]
        chunk = match["chunk"]
        indicator_name = indicator_label.split(" — ")[0]

        output[ind_id] = {
            "indicator_name": indicator_name,
            "match_level": _score_to_match_level(score),
            "confidence": round(score, 4),
            "exact_quote": chunk.text,
            "page_number": chunk.page_number,
        }

    _save_results(country, output)
    return output

# Results persistence
def _save_results(country: str, results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / f"{country.lower()}.json"
    dest.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def load_results(country: str) -> Optional[dict]:
    """Load previously saved results for a country. Returns None if not found."""
    dest = RESULTS_DIR / f"{country.lower()}.json"
    if dest.exists():
        try:
            return json.loads(dest.read_text())
        except Exception:
            return None
    return None


def load_all_results() -> dict[str, dict]:
    """Load all saved results. Returns dict mapping country name to results."""
    from crawler import SOURCES
    all_results = {}
    for country in SOURCES:
        result = load_results(country)
        if result is not None:
            all_results[country] = result
    return all_results


if __name__ == "__main__":
    import sys
    from extractor import extract_text
    from crawler import SOURCES, _dest_path

    target = sys.argv[1] if len(sys.argv) > 1 else None
    countries = [target] if target else list(SOURCES.keys())

    for country in countries:
        pdf_path = _dest_path(country)
        if not pdf_path.exists():
            print(f"No PDF for {country}, skipping")
            continue

        print(f"\nExtracting {country}...")
        extraction = extract_text(str(pdf_path))
        if extraction.error:
            print(f"  Extraction error: {extraction.error}")
            continue

        print(f"  {len(extraction.chunks)} chunks extracted")
        print(f"  Running AI analysis...")

        results = map_chunks(
            extraction.chunks,
            country,
            progress_callback=lambda p: print(f"  {p*100:.0f}%", end="\r"),
        )

        print(f"\n  Matched {len(results)} indicators:")
        for ind_id, match in sorted(results.items()):
            print(f"  [{match['match_level']:9s}] {ind_id}: {match['indicator_name']} ({match['confidence']:.0%})")
