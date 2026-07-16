"""
app/api/health.py
==================
GET /health — liveness + readiness check.
"""

from __future__ import annotations

from fastapi import APIRouter
from app.models import HealthResponse
from app.dependencies import get_claims_df, get_rag_chain

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Returns service health status including:
    - Whether claims are loaded
    - Whether the FAISS index is available
    - Current embedding model name
    """
    from pathlib import Path
    from config.settings import app_cfg

    df = get_claims_df()
    index_ready = (app_cfg.faiss_index_path / "faiss.index").exists()

    return HealthResponse(
        status="ok",
        pipeline_ready=not df.empty and index_ready,
        claims_loaded=len(df),
        index_loaded=index_ready,
        model=getattr(app_cfg, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )
