# 🔍 AuditLens — ESG Audit Assistant

> **AI-powered end-to-end ESG audit pipeline** | Fine-tuned FinBERT · SHAP · RAG · GCP Cloud Run · GitHub Actions CI/CD

[![Deploy](https://github.com/your-username/esg-audit-assistant/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-username/esg-audit-assistant/actions/workflows/deploy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 One-Line Pitch

> "I built a system that ingests live SEC ESG filings daily, extracts sustainability claims, classifies each using a fine-tuned FinBERT model with SHAP explanations, retrieves relevant GRI/TCFD/SASB/ISSB standards via RAG, performs evidence-based consistency and compliance checks against historical filings and CDP data, and generates structured audit observations — deployed on GCP Cloud Run with GitHub Actions CI/CD and monitored retraining on data drift."

---

## 🏗️ Pipeline Architecture

The AuditLens pipeline consists of six core stages:

```text
1. EDGAR Ingestion
       ↓
2. PDF/HTML Parsing (pages, sections, tables, metadata)
       ↓
3. Claim Extraction (Sentence Splitter, Optional keyword pre-filter, FinBERT ESG filter, Gemini atomic extraction, Quality filter)
       ↓
4. VerificationPipeline (L1 Internal, L2 Historical, L3 Standards)
       ↓
5. Aggregation
       ↓
6. ReportGenerator
```

---

## 🔍 Core Pipeline Chronology

### 1. EDGAR Ingestion
* **Trigger**: A stock ticker (e.g. `AAPL`) is mapped to its central Index Key (CIK).
* **Filing Retrieval**: Resolves and downloads the latest available SEC 10-K XHTML/HTML document, saving it locally to `data/raw/<TICKER>/`.

### 2. Document Parsing
* **XHTML/HTML Processing**: Cleans and strips styling tags, formatting raw visible text.
* **Structural Parsing**: Produces an `ExtractedDocument` object holding:
  * **Pages**: Raw text segmented into standard page-like intervals.
  * **Sections**: Document contents classified into key sectors (e.g. `environmental`, `social`, `governance`, `general`).
  * **Tables**: Extracted tabular structures formatted as pipe-separated content for reference.
  * **Metadata**: Company details and filing period metadata.

### 3. Claim Extraction
* **Sentence Splitting**: spaCy breaks section blocks into individual sentences.
* **Optional Keyword Pre-filtering**: A lightweight, optional keyword filter prunes clearly irrelevant sentences to reduce downstream GPU/CPU inference costs.
* **FinBERT ESG Classifier**: Sentences are routed through `yiyanghkust/finbert-esg` to classify ESG relevance.
* **Gemini Atomic Extraction**: Relevant sentences are passed to Gemini, which splits compound statements into atomic assertions, categorizing by type (`quantitative`, `commitment`, `compliance`).
* **Evidence Preservation**: Every atomic claim retains its original parent sentence, surrounding context, start/end character offsets, and normalized text.
* **Post-Extraction Quality Filter**: A dedicated filter prunes duplicate, non-auditable, philosophical, generic marketing, or purely operational disclosures.

### 4. Verification Pipeline
* **L1 Internal Check**: Verifies if the numeric assertions in a claim match tables or disclosures elsewhere inside the *same* document.
* **L2 Historical Check**: Compares claims to previous years' filings to identify baseline or metric contradictions.
* **L3 Standards Check**: Queries the FAISS vector database containing disclosure framework indexes, evaluating the claim against retrieved standards to identify missing or incomplete required disclosures.

### 5. Verdict Aggregation
* **Risk Scoring**: Combines L1, L2, and L3 verification metrics into a consolidated Audit Risk Score.
* **Verdict Assignment**: Maps scores to an actionable auditing category: `CONSISTENT`, `PARTIALLY_CONSISTENT`, `INCONSISTENT`, or `HIGH_RISK`.

### 6. Audit Report Generation
* **Report Compilation**: The `ReportGenerator` compiles verification verdicts.
* **RAG Observations**: Generates structured audit notes, recommended actions, and specific framework citations using retrieved standards context for non-compliant claims.
* **Persistence**: Outputs a consolidated audit report in JSON (`audit_report.json`) and Markdown formats.

---

## 📁 Project Structure

```
esg-audit-assistant/
├── config/
│   ├── settings.py              # Pydantic env var config
│   └── logging_config.yaml      # Structured JSON logging (GCP compatible)
├── src/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py     # EDGAR REST API → daily PDF downloads
│   │   └── cdp_fetcher.py       # CDP API stub (Level 4 check)
│   ├── extraction/
│   │   ├── pdf_extractor.py     # pdfplumber text + table extraction
│   │   └── claim_detector.py    # spaCy + regex ESG claim detection
│   ├── classification/
│   │   ├── finbert_classifier.py # FinBERT E/S/G + risk score
│   │   └── shap_explainer.py    # Token-level SHAP explanations
│   ├── consistency/
│   │   ├── internal_checker.py  # L1: same-document table check
│   │   ├── historical_checker.py # L2: prior year EDGAR cross-check
│   │   ├── standard_checker.py  # L3: GRI/TCFD/ISSB compliance check
│   │   ├── cdp_checker.py       # L4: CDP submission cross-check (optional)
│   │   └── aggregator.py        # Weighted verdict aggregation
│   ├── rag/
│   │   ├── embedder.py          # all-MiniLM-L6-v2 (CPU)
│   │   ├── vector_store.py      # FAISS index build/search
│   │   └── rag_chain.py         # LangChain + Gemini audit observation
│   ├── mlops/
│   │   ├── drift_monitor.py     # Evidently AI distribution monitoring
│   │   └── retrain_trigger.py   # GitHub Actions workflow dispatch
│   └── dashboard/
│       ├── app.py               # Streamlit main app (7 pages)
│       └── components/          # Reusable UI components
├── notebooks/
│   └── finbert_finetune.py      # Kaggle T4 GPU fine-tuning script
├── scripts/
│   └── build_index.py           # One-time FAISS index builder
├── tests/                       # pytest suite (mocked — no GPU needed)
├── .github/workflows/
│   ├── deploy.yml               # CI/CD: test → build → Cloud Run
│   └── retrain.yml              # Monitored retraining on drift detection
├── Dockerfile                   # Multi-stage production image
├── docker-compose.yml           # Local dev setup
├── dvc.yaml                     # Data pipeline stages
└── requirements.txt             # All Python dependencies
```

---

## 🚀 Quick Start (Local)

### 1. Clone and setup environment

```bash
git clone https://github.com/your-username/esg-audit-assistant.git
cd esg-audit-assistant

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment variables

```bash
copy .env.example .env
# Open .env and fill in:
#   GEMINI_API_KEY=your_key_from_aistudio.google.com
#   EDGAR_USER_AGENT=YourName your@email.com
```

### 3. Download regulatory standard PDFs (Week 3)

Place these PDFs in `data/knowledge_base/`:

| Standard | Download URL |
|----------|-------------|
| GRI 305 (Emissions) | https://www.globalreporting.org/standards/ |
| TCFD Recommendations | https://www.fsb-tcfd.org/publications/ |
| SASB Standards | https://sasb.org/standards/download/ |
| ISSB (IFRS S1/S2) | https://www.ifrs.org/groups/international-sustainability-standards-board/ |
| EU CSRD | https://eur-lex.europa.eu |

Then build the FAISS index:
```bash
python scripts/build_index.py
```

### 4. Run the dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

### 5. (Optional) Run with Docker

```bash
docker-compose up --build
# Opens at http://localhost:8501
```

---

## 🎯 Week-by-Week Build Guide

### Week 1 — Data Ingestion + Extraction
```bash
# Test EDGAR fetcher
python -c "
from src.ingestion.edgar_fetcher import EdgarFetcher
f = EdgarFetcher()
filings = f.fetch_recent_filings('AAPL', days_back=365)
print(f'Found {len(filings)} filings')
"

# Test claim detector
python -c "
from src.extraction.claim_detector import ClaimDetector
d = ClaimDetector()
claims = d.detect_claims('We reduced Scope 1 emissions by 40% since 2019.')
print(claims[0].text, claims[0].confidence)
"
```

### Week 2 — FinBERT Fine-Tuning on Kaggle
1. Go to [kaggle.com](https://kaggle.com) → New Notebook → Enable GPU T4
2. Upload `notebooks/finbert_finetune.py` or paste cells
3. Add secrets: `HF_TOKEN`, `HF_USERNAME` in Kaggle secrets
4. Run all cells — MLflow logs both pretrained vs fine-tuned runs
5. After upload, update `.env`: `FINBERT_MODEL_ID=your-username/finbert-esg`

### Week 3 — RAG Pipeline
```bash
# After downloading standard PDFs:
python scripts/build_index.py

# Test RAG chain
python -c "
from src.rag.vector_store import VectorStore
from config.settings import app_cfg
store = VectorStore.load(app_cfg.faiss_index_path)
results = store.search('Scope 1 GHG emission disclosure requirements')
print(results[0].chunk.text[:200])
"
```

### Week 4 — Dashboard + SHAP
```bash
streamlit run src/dashboard/app.py
# Upload Apple's 2023 10-K PDF → verify claims appear + SHAP chart renders
```

### Week 5 — Drift Monitoring
```bash
# Test Evidently drift check
python -c "
import pandas as pd
from src.mlops.drift_monitor import DriftMonitor
monitor = DriftMonitor()
ref_df = pd.DataFrame({'text': ['claim 1', 'claim 2'], 'confidence': [0.8, 0.7], 'risk_score': [0.3, 0.4], 'esg_label': ['E', 'S']})
monitor.set_reference(ref_df)
report = monitor.check_drift(ref_df)
print('Drift detected:', report.drift_detected)
"
```

### Week 6 — Docker + GCP Deployment
See [GCP Setup Guide](#-gcp-setup-step-by-step) below.

---

## ☁️ GCP Setup Step-by-Step

### Prerequisites
- Google account
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed

### Step 1: Create GCP Project
```bash
# Install gcloud SDK first: https://cloud.google.com/sdk/docs/install
gcloud projects create auditlens-PROJECT_ID --name="AuditLens"
gcloud config set project auditlens-PROJECT_ID
gcloud billing projects link auditlens-PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
# ⚠️ SET BILLING ALERT AT $1 IMMEDIATELY in GCP Console → Billing → Budgets
```

### Step 2: Enable APIs
```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  storage.googleapis.com
```

### Step 3: Create Artifact Registry + Storage Bucket
```bash
gcloud artifacts repositories create auditlens-registry \
  --repository-format=docker \
  --location=us-central1

gsutil mb -l us-central1 gs://auditlens-storage
```

### Step 4: Create Service Account for GitHub Actions
```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions SA"

# Grant required permissions
for role in roles/run.admin roles/artifactregistry.writer roles/storage.admin; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role=$role
done

# Export key (upload to GitHub Secrets as GCP_SA_KEY)
gcloud iam service-accounts keys create gcp-sa-key.json \
  --iam-account=github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### Step 5: Add GitHub Secrets
In your GitHub repo → Settings → Secrets → Actions, add:

| Secret | Value |
|--------|-------|
| `GCP_SA_KEY` | Contents of `gcp-sa-key.json` |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_BUCKET` | `auditlens-storage` |
| `GEMINI_API_KEY` | Your Gemini API key |
| `HF_TOKEN` | Your HuggingFace token |
| `HF_USERNAME` | Your HuggingFace username |
| `EDGAR_USER_AGENT` | `YourName your@email.com` |

### Step 6: Set up Cloud Scheduler (daily ingestion)
```bash
gcloud scheduler jobs create http daily-edgar-ingestion \
  --schedule="0 6 * * *" \
  --uri="https://your-cloud-run-url.run.app/ingest" \
  --http-method=POST \
  --location=us-central1 \
  --message-body='{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

### Step 7: Deploy
```bash
git push origin main
# GitHub Actions will automatically build + deploy
# Watch progress at: github.com/your-username/esg-audit-assistant/actions
```

---

## 🔬 Running Tests

```bash
# Run all tests (no GPU needed — models are mocked)
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_claim_detector.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

---

## 🧠 Model Information

### FinBERT (Classification)
- **Pretrained base**: `ProsusAI/finbert` (pre-trained on financial text)
- **Fine-tuned**: `your-username/finbert-esg` (trained on FinESG + Climate FEVER)
- **Task**: Multi-class classification → E, S, or G category + risk score
- **Training**: Kaggle T4 GPU, ~3 epochs, logged in MLflow
- **Fallback**: Uses pretrained version during Week 1–2 before fine-tuning

### Embeddings (RAG)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Runs on**: CPU — no GPU needed for inference

### Gemini 1.5 Flash (Generation)
- **Purpose**: Structured audit observation generation
- **Hallucination control**: RAG-grounded — only generates from retrieved standard chunks
- **Cost**: Free tier (Google AI Studio)

---

## 📋 Consistency Check Levels

| Level | Check | Method |
|-------|-------|--------|
| L1 | Internal document consistency | pdfplumber table search in same PDF |
| L2 | Historical cross-check | EDGAR prior year filing comparison |
| L3 | Standard compliance (RAG) | FAISS retrieval + GRI/TCFD rule matching |
| L4 | CDP cross-check (optional) | CDP API submission comparison |

**Verdict scale**: `CONSISTENT` → `PARTIALLY_CONSISTENT` → `INCONSISTENT` → `HIGH_RISK`

---

## 🔄 Retraining Pipeline

```
New filings daily → Evidently AI monitors distribution
                          ↓
               Drift score > 0.3 threshold?
               NO  → continue as normal
               YES → GitHub Actions retrain.yml dispatched
                          ↓
                  Retrain FinBERT on Kaggle
                          ↓
               Evaluate: new F1 > existing F1?
               YES → upload to HF Hub → redeploy Cloud Run
               NO  → keep existing model → alert raised
```

---

## 💰 Cost Summary

| Service | Usage | Cost |
|---------|-------|------|
| Kaggle GPU | FinBERT training | **Free** |
| HuggingFace Hub | Model storage | **Free** |
| GCP Cloud Run | App hosting | **Free tier** |
| GCP Cloud Storage | Files + FAISS | **Free** (5GB) |
| GCP Artifact Registry | Docker images | **Free** (0.5GB) |
| GCP Cloud Scheduler | Daily cron | **Free** (3 jobs) |
| GitHub Actions | CI/CD | **Free** |
| Gemini API | Generation | **Free tier** |

> **⚠️ Set a GCP billing alert at $1 immediately after creating your project.**

---

## 🎤 Interview Narrative

> "I built an end-to-end ESG audit assistant that ingests live SEC EDGAR filings daily, extracts sustainability claims using NLP, and classifies each using a fine-tuned FinBERT transformer with SHAP explainability — so auditors can see exactly why a claim was flagged. The system performs evidence-based consistency and compliance checks at four levels: internal document consistency, historical cross-referencing against previous EDGAR filings, GRI, TCFD, and ISSB standard compliance via RAG, and optional CDP cross-validation. Gemini generates structured audit observations grounded in retrieved standard text to reduce hallucination risk. The whole pipeline is containerized with Docker, deployed on GCP Cloud Run, with GitHub Actions CI/CD and Evidently AI drift monitoring — which triggers a monitored retraining workflow on Kaggle when filing patterns shift, with a model evaluation gate before any new deployment."

---

## 📚 Key References

- [SEC EDGAR REST API](https://www.sec.gov/developer)
- [GRI Standards](https://www.globalreporting.org/standards/)
- [TCFD Recommendations](https://www.fsb-tcfd.org/publications/)
- [ProsusAI/finbert on HuggingFace](https://huggingface.co/ProsusAI/finbert)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [LangChain + Gemini](https://python.langchain.com/docs/integrations/llms/google_ai)

---

*Built for PwC ESG audit fresher interview — demonstrating end-to-end ML engineering with production deployment.*
