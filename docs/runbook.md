# Runbook — UN Hackathon RDTII Analyzer

Quick reference: symptom → diagnose → fix. Each section maps to a rootcause JSON in `docs/rootcause/`.

---

## §1 — World Map crashes: `ModuleNotFoundError: No module named 'pydeck'`

**Rootcause:** `pydeck-missing-requirements`

**Diagnose:** Run `pip show pydeck` — if not found, this is the issue.

**Fix:**
```bash
# Add to un-rdtii/requirements.txt
echo "pydeck" >> un-rdtii/requirements.txt
pip install pydeck
```

---

## §2 — Comparison Table crashes: `AttributeError: 'Styler' object has no attribute 'applymap'`

**Rootcause:** `applymap-pandas-deprecated`

**Diagnose:** Run `pip show pandas` — if version >= 2.1, this is the issue.

**Fix:** In `un-rdtii/app.py:584`, change:
```python
# Before
styled = df.style.applymap(_style_cell, subset=country_cols)
# After
styled = df.style.map(_style_cell, subset=country_cols)
```

---

## §3 — OCR fallback returns empty text on scanned PDFs

**Rootcause:** `surya-ocr-api-breakage`

**Diagnose:** Run `pip show surya-ocr` to check the version. Then test:
```python
from surya.ocr import run_ocr
from surya.model_lst import ModelLst
```
If either import fails or `run_ocr` signature has changed, this is the issue.

**Fix:** Check the installed surya-ocr changelog for the current API. Update `un-rdtii/extractor.py:60-74` to match the current `run_ocr()` signature and model loading pattern. Pin the version in `requirements.txt` once confirmed working.

---

## §4 — Sidebar shows Tesseract warning even though Surya OCR is installed

**Rootcause:** `tesseract-misleading-warning`

**Diagnose:** Warning always appears regardless of environment — it is dead code.

**Fix:** In `un-rdtii/app.py`, remove:
- Lines 58–61: `_check_tesseract()` function and `_tesseract_ok` call
- The sidebar block that checks `_tesseract_ok` and shows the warning (around line 110)

---

## §5 — README says `facebook/bart-large-mnli` but wrong model loads

**Rootcause:** `readme-model-name-mismatch`

**Diagnose:** Check `un-rdtii/mapper.py:32` — `MODEL_NAME = "cross-encoder/nli-distilroberta-base"`.

**Fix:** Update `README.md` — replace all occurrences of `facebook/bart-large-mnli` with `cross-encoder/nli-distilroberta-base`. The model size (~500 MB) is still correct.

---

## §6 — CLI `/compare` command does nothing

**Diagnose:** Known unimplemented feature — `datatrade-cli/main.py:352` prints "not yet implemented".

**Fix:** Implement `_cmd_compare()` in `main.py` — run FAISS search per indicator (10 indicators), score each country, print a comparison table using `rich`.

---

## §8 — `/ask` unexpectedly triggers Thailand crawl on unrelated queries

**Rootcause:** `detect-country-false-positive`

**Diagnose:** Query containing "with" (e.g. "what rules apply with data transfers?") triggers TH country detection. Seen in pipeline logs as `lazy: TH cache MISS — no source URLs configured`.

**Fix:** Already applied — removed `"th "` from `_detect_country()` keyword list in `datatrade-cli/modules/pipeline.py`. Thailand is covered by `"thailand"`, `"thai"`, `"ไทย"`, `"pdpa"`.

---

## §7 — `playwright install` not documented, web crawl fails

**Diagnose:** `datatrade-cli` crawler uses Playwright but the install step is missing from docs.

**Fix:** After `pip install -r requirements.txt`, run:
```bash
playwright install chromium
```
Add this step to `datatrade-cli/PROTOTYPE.md` under Environment Setup.
