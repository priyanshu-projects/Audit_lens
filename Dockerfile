# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Installs all Python packages. Kept separate to keep final image lean.
FROM python:3.11-slim AS builder

WORKDIR /app

# Build-time system deps (compilers for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — layer cache: only re-runs if requirements.txt changes
COPY requirements.txt .

# Install all Python packages into /install (copied to final stage)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Download spaCy model into /install so it travels to the final stage
RUN pip install --prefix=/install spacy && \
    PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_sm --prefix /install || true


# ── Stage 2: Production image ─────────────────────────────────────────────────
# Lean runtime image — no compilers, no build tools.
FROM python:3.11-slim AS production

WORKDIR /app

# Runtime system deps only (libgomp1 needed by FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from builder stage
COPY --from=builder /install /usr/local

# ── Bake ML models into the image ────────────────────────────────────────────
# Models download HERE during docker build, not at container startup.
# This makes cold starts fast — no 5-minute model download on first request.

# Point all HuggingFace/sentence-transformers caches to /app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# Download FinBERT ESG classifier (yiyanghkust/finbert-esg — ~440 MB)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('yiyanghkust/finbert-esg'); \
AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg'); \
print('FinBERT downloaded OK')"

# Download sentence-transformers embedding model (~90 MB)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Embedding model downloaded OK')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/       ./src/
COPY config/    ./config/
COPY app/       ./app/
COPY scripts/   ./scripts/

# Copy pre-built FAISS knowledge base (GRI + SASB + TCFD standards)
# faiss.index + chunks.json + PDFs are already built locally — no rebuild needed
COPY data/knowledge_base/ ./data/knowledge_base/

# Create writable dirs for runtime temp files (uploaded PDFs, reports)
RUN mkdir -p /tmp/auditlens_uploads /tmp/auditlens_reports

# ── Security: run as non-root user ───────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Environment variables ─────────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV LOG_LEVEL=INFO
ENV FINBERT_MODEL_ID=yiyanghkust/finbert-esg
ENV EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8501 = Streamlit (auditlens-ui service)
# 8080 = FastAPI  (auditlens-api service) — override CMD when deploying API
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# Give 120s start-period because models need to load into memory on first start
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command: Streamlit dashboard ──────────────────────────────────────
# When deploying the FastAPI service, override with:
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
