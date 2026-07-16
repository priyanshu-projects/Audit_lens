"""
app/api/query.py
=================
POST /query — RAG-powered question answering over the knowledge base.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models import QueryRequest, QueryResponse
from app.dependencies import get_rag_chain, get_claims_df

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a natural-language question. Returns an answer grounded in retrieved
    document chunks with citations and a confidence level.

    Supports queries like:
    - "What emission reduction targets has Apple committed to?"
    - "Is single-use plastic reduction mentioned?"
    - "Compare ESG commitments between Apple 2024 and Apple 2025."
    """
    rag_chain = get_rag_chain()
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not available. Run 'dvc repro' first to build the index.",
        )

    try:
        result = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            filters=request.filters,
        )
    except Exception as exc:
        logger.error(f"RAG query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Estimate confidence from retrieval scores
    scores = [c.get("score", 0.0) for c in result.get("source_documents", [])]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if avg_score >= 0.75:
        confidence = "High"
    elif avg_score >= 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    sources = [
        f"{c.get('metadata', {}).get('document', 'Unknown')} — Page {c.get('metadata', {}).get('page', '?')}"
        for c in result.get("source_documents", [])
    ]

    retrieved_chunks = [
        {
            "text": c.get("page_content", ""),
            "score": c.get("score", 0.0),
            "metadata": c.get("metadata", {}),
        }
        for c in result.get("source_documents", [])
    ] if request.filters and request.filters.get("explainability") == "true" else None

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", "The documents do not contain enough information."),
        confidence=confidence,
        sources=list(dict.fromkeys(sources)),   # deduplicate preserving order
        retrieved_chunks=retrieved_chunks,
    )
