"""
app/main.py
============
AuditLens FastAPI application entrypoint.

Start with:
    uvicorn app.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import claims, documents, health, query, report, verify
from app.dependencies import get_claims_df, get_documents, get_finbert, get_rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: eagerly load heavy singletons (FinBERT, claims, FAISS index)
    so the first request is not slow.
    """
    logger.info("AuditLens API starting up — loading models & data...")
    get_finbert()         # load FinBERT into GPU/CPU memory
    get_claims_df()       # load claims.parquet
    get_documents()       # load documents.json
    get_rag_chain()       # load FAISS index + Gemini chain
    logger.success("Startup complete. All singletons loaded.")
    yield
    logger.info("AuditLens API shutting down.")


app = FastAPI(
    title="AuditLens API",
    description=(
        "Intelligent ESG Audit Assistant — extracts, verifies, and queries "
        "ESG claims from corporate filings using RAG + FinBERT + Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(claims.router)
app.include_router(verify.router)
app.include_router(report.router)
app.include_router(documents.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "AuditLens API",
        "version": "1.0.0",
        "docs": "/docs",
    }
