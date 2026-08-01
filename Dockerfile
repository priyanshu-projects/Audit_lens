# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Installs all Python packages. Re-runs only when requirements.txt changes.
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


# ── Stage 2: Model Cache ──────────────────────────────────────────────────────
# Downloads heavy ML models INDEPENDENTLY of requirements.txt.
# This layer is cached even when requirements.txt changes — saves ~530 MB re-download.
FROM python:3.11-slim AS model-cache

# Install only what's needed to download the models
RUN pip install --no-cache-dir \
    "transformers>=4.35.0,<4.40.0" \
    torch \
    sentence-transformers

# Cache directory baked into the image
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

WORKDIR /app

# Download FinBERT ESG classifier (~440 MB)
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


# ── Stage 3: Production image ─────────────────────────────────────────────────
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

# Copy pre-baked ML models from the independent model-cache stage
COPY --from=model-cache /app/models /app/models

# ── Environment variables ─────────────────────────────────────────────────────
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
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

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/       ./src/
COPY config/    ./config/
COPY app/       ./app/
COPY scripts/   ./scripts/

# Copy pre-built FAISS knowledge base (GRI + SASB + TCFD standards)
COPY data/knowledge_base/ ./data/knowledge_base/

# Create writable dirs for runtime temp files and data outputs
RUN mkdir -p /app/data/raw /app/data/classified_runs /app/data/processed /tmp/auditlens_uploads /tmp/auditlens_reports

# ── Security: run as non-root user ───────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app /tmp/auditlens_uploads /tmp/auditlens_reports
USER appuser

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command: Streamlit dashboard ──────────────────────────────────────
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
