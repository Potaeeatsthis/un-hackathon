"""
mapper.py — AI mapping to RDTII indicators using Hugging Face zero-shot classification
Uses facebook/bart-large-mnli to map text chunks to indicators without any API key.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

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

MODEL_NAME = "cross-encoder/nli-distilroberta-base"
CONFIDENCE_THRESHOLD = 0.5
PRIMARY_THRESHOLD = 0.85
CONTEXTUAL_THRESHOLD = 0.70
BATCH_SIZE = 8
CANDIDATE_COUNT = 3

# Module-level cache so the model is only loaded once per process
_classifier = None


# Helper functions
def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = CrossEncoder(MODEL_NAME)
    return _classifier

def _score_to_match_level(score: float) -> str:
    if score >= PRIMARY_THRESHOLD:
        return "Primary"
    if score >= CONTEXTUAL_THRESHOLD:
        return "Contextual"
    if score >= CONFIDENCE_THRESHOLD:
        return "Implicit"
    return "No match"


def _entailment_scores(raw_scores, classifier) -> np.ndarray:
    """Convert NLI logits to entailment probabilities."""
    scores = np.asarray(raw_scores, dtype=float)

    if scores.ndim == 1:
        # Some CrossEncoder models return one logit per pair. Keep this fallback
        # so the mapper remains usable if MODEL_NAME changes later.
        return 1 / (1 + np.exp(-scores))

    if scores.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {scores.shape}")

    id2label = getattr(classifier.model.config, "id2label", {})
    entailment_idx = None
    for idx, label in id2label.items():
        if str(label).lower() == "entailment":
            entailment_idx = int(idx)
            break

    if entailment_idx is None:
        entailment_idx = 1

    if entailment_idx >= scores.shape[1]:
        raise ValueError(
            f"Entailment label index {entailment_idx} is outside model output shape {scores.shape}"
        )

    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probabilities[:, entailment_idx]


def _remember_candidate(candidates: dict[str, list[dict]], ind_id: str, score: float, chunk: TextChunk) -> None:
    candidates.setdefault(ind_id, []).append({"score": float(score), "chunk": chunk})
    candidates[ind_id].sort(key=lambda item: item["score"], reverse=True)
    del candidates[ind_id][CANDIDATE_COUNT:]


def _format_candidate_output(candidates: dict[str, list[dict]]) -> dict[str, list[dict]]:
    formatted: dict[str, list[dict]] = {}
    for ind_id, matches in candidates.items():
        indicator_name = INDICATORS[ind_id].split(" — ")[0]
        formatted[ind_id] = [
            {
                "indicator_name": indicator_name,
                "review_label": "Candidate evidence for review",
                "confidence": round(item["score"], 4),
                "exact_quote": item["chunk"].text,
                "page_number": item["chunk"].page_number,
            }
            for item in matches
        ]
    return formatted

# Main functions

def map_chunks_with_candidates(
    chunks: list[TextChunk],
    country: str,
    progress_callback=None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
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
    candidates: dict[str, list[dict]] = {}

    total = len(chunks)
    processed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]

        try:
            # Run classification for each text in the batch
            for chunk in batch:
                # Create pairs: (text, label) for each candidate label
                pairs = [(chunk.text, label) for label in candidate_labels]
                
                # NLI CrossEncoder returns logits for contradiction, entailment, neutral.
                # Use entailment probability as the confidence score.
                scores = _entailment_scores(classifier.predict(pairs), classifier)
                
                for label_idx, score in enumerate(scores):
                    ind_id = indicator_ids[label_idx]
                    _remember_candidate(candidates, ind_id, float(score), chunk)

                    if score < CONFIDENCE_THRESHOLD:
                        continue

                    if ind_id not in best_match or score > best_match[ind_id]["score"]:
                        best_match[ind_id] = {"score": float(score), "chunk": chunk}

                processed += 1
                if progress_callback:
                    progress_callback(processed / total)

        except Exception as exc:
            print(f"\n  Mapper error in batch starting at chunk {batch_start}: {exc}")
            raise

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
    return output, _format_candidate_output(candidates)


def map_chunks(
    chunks: list[TextChunk],
    country: str,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Run zero-shot classification and return confirmed matches only.

    Use map_chunks_with_candidates() when low-confidence review candidates are needed.
    """
    output, _candidates = map_chunks_with_candidates(chunks, country, progress_callback)
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

        results, candidates = map_chunks_with_candidates(
            extraction.chunks,
            country,
            progress_callback=lambda p: print(f"  {p*100:.0f}%", end="\r"),
        )

        print(f"\n  Matched {len(results)} indicators:")
        for ind_id, match in sorted(results.items()):
            print(f"  [{match['match_level']:9s}] {ind_id}: {match['indicator_name']} ({match['confidence']:.0%})")

        print(f"\n  Candidate evidence for review:")
        for ind_id, matches in sorted(candidates.items()):
            low_confidence = [
                match for match in matches
                if match["confidence"] < CONFIDENCE_THRESHOLD
            ][:CANDIDATE_COUNT]
            if not low_confidence:
                continue
            print(f"  {ind_id}: {INDICATORS[ind_id].split(' — ')[0]}")
            for idx, match in enumerate(low_confidence, start=1):
                snippet = match["exact_quote"].replace("\n", " ")[:220]
                suffix = "..." if len(match["exact_quote"]) > 220 else ""
                print(
                    f"    Candidate {idx}: {match['confidence']:.0%}, "
                    f"page {match['page_number']} — {snippet}{suffix}"
                )
