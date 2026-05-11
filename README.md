# RDTII Regulatory Analyzer

**AI-powered digital trade regulation mapping for Asia-Pacific economies**

Built for the [UN ESCAP Hackathon on Digital Trade Governance](https://www.unescap.org/).

---

## What This Tool Does

The RDTII Regulatory Analyzer automatically downloads, extracts, and analyzes national digital trade regulations from Thailand, Vietnam, and Indonesia. It maps regulatory text to the 10 indicators of the **Regulatory & Digital Trade Indicators Index (RDTII)** — specifically Pillars 6 (Cross-Border Data Flows) and 7 (Data Protection & Cybersecurity).

The AI analysis runs entirely locally using [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) — no API key, no paid subscription, no internet connection required after the model and documents are downloaded once.

---

## Why It Matters

Digital trade governance is increasingly shaped by national data protection laws that create barriers or enablers for cross-border data flows. For Asia-Pacific economies, understanding where each country stands on these 10 indicators is critical for:

- Negotiating regional digital trade agreements
- Identifying regulatory gaps and harmonization opportunities
- Supporting ASEAN and UN ESCAP policy dialogue
- Reducing compliance uncertainty for businesses operating across borders

---

## RDTII Framework — Pillars 6 and 7

### Pillar 6: Cross-Border Data Flows
| ID  | Indicator |
|-----|-----------|
| 6.1 | Ban and local processing requirements |
| 6.2 | Local storage requirements |
| 6.3 | Infrastructure requirements |
| 6.4 | Conditional flow regimes |
| 6.5 | Not participating in data transfer agreements |

### Pillar 7: Data Protection & Cybersecurity
| ID  | Indicator |
|-----|-----------|
| 7.1 | Lack of comprehensive data protection framework |
| 7.2 | Lack of dedicated cybersecurity framework |
| 7.3 | Data retention requirements |
| 7.4 | DPIA or DPO requirements |
| 7.5 | Government access to personal data |

---

## Countries Covered

| Country | Document | Reason |
|---------|----------|--------|
| **Thailand** | Personal Data Protection Act B.E. 2562 (2019) | First ASEAN country with GDPR-aligned comprehensive PDPA |
| **Vietnam** | Decree 13/2023 on Personal Data Protection | Strong data localization provisions; major Southeast Asian digital economy |
| **Indonesia** | Government Regulation PP 71/2019 | Largest ASEAN economy; strategic sector localization requirements |

---

## Installation

### 1. Clone or download this project

```bash
git clone <repo-url>
cd rdtii-analyzer
```

### 2. Create a Python virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate.bat     # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install poppler (for Surya OCR)

Poppler is required for PDF-to-image conversion used by Surya OCR:

**Ubuntu / Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS (Homebrew):**
```bash
brew install poppler
```

**Windows:**
Download from https://github.com/oschwartz10612/poppler-windows/releases/ and add the `bin/` folder to your `PATH`.

Note: Tesseract is no longer needed; Surya provides better OCR accuracy for scanned documents.

---

## Running the App

```bash
cd un-rdtii
streamlit run app.py
```

The app opens at http://localhost:8501

**On first run:**
- The CrossEncoder model (~500 MB) will be downloaded and cached
- Surya OCR components will be initialized (required for scanned PDFs)
- Subsequent runs will be instant as models are cached locally

---

## How the AI Works

The tool uses **Natural Language Inference (NLI)** via the `cross-encoder/nli-distilroberta-base` model from Hugging Face:

1. PDF text is split into ~500-character chunks with 50-character overlap.
2. Each chunk is paired with all 10 indicator descriptions.
3. The CrossEncoder model scores each (text, indicator) pair for semantic entailment.
4. Scores are normalized to 0–1 and only kept above **0.5** confidence:
   - **Primary** (>0.85) — direct, explicit regulatory text
   - **Contextual** (0.70–0.85) — related provisions that imply the indicator
   - **Implicit** (0.50–0.70) — indirect or partial coverage

CrossEncoders are more accurate than pipeline-based models for NLI because they jointly score (premise, hypothesis) pairs rather than independently classifying text. This approach captures semantic relationships more precisely.

The model is downloaded from Hugging Face on first run (~500 MB) and cached locally.
---

## Project Structure

```
un-rdtii/
├── app.py              # Streamlit web interface with 4-page navigation
├── crawler.py          # PDF downloader with fallback domain crawling
├── extractor.py        # PDF text extraction + Surya OCR fallback
├── mapper.py           # NLI-based indicator classification
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── data/
    ├── documents/      # Downloaded PDFs (auto-created)
    └── results/        # JSON analysis results (auto-created)
```

## Acknowledgements

- [UN ESCAP](https://www.unescap.org/) — for organizing the hackathon and the RDTII framework
- [Hugging Face](https://huggingface.co/facebook/bart-large-mnli) — for the `facebook/bart-large-mnli` model
- [Streamlit](https://streamlit.io/) — for the web framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) — for PDF extraction
