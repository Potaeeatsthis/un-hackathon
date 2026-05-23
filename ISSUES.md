# Issues

## Bugs (blocking — fix before demo)

- [x] `pydeck` missing from `un-rdtii/requirements.txt` — World Map page crashes on fresh install → runbook §1
- [x] `df.style.applymap()` deprecated in pandas 2.1+ — Comparison Table crashes → runbook §2
- [x] Surya OCR API likely broken on modern `surya-ocr` versions — scanned PDF pages return empty text → runbook §3

## Bugs (non-blocking)

- [x] Tesseract warning in sidebar is misleading — app uses Surya OCR, not Tesseract → runbook §4
- [x] README says `facebook/bart-large-mnli` but actual model is `cross-encoder/nli-distilroberta-base` → runbook §5
- [x] `playwright install chromium` step missing from `datatrade-cli/PROTOTYPE.md` → runbook §7

## Feature Gaps

- [ ] `datatrade-cli` `/compare` command not implemented → runbook §6
- [ ] `datatrade-cli` `.env.example` exists but `config.py` does not load from `.env` — env vars are hardcoded in config

## Done

<!-- Move items here when fixed -->
