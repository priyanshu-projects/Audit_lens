from __future__ import annotations
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models import QueryRequest, QueryResponse
from app.dependencies import get_rag_chain
router = APIRouter(prefix='/query', tags=['Query'])

@router.post('', response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    rag_chain = get_rag_chain()
    if rag_chain is None:
        raise HTTPException(status_code=503, detail='RAG chain not available. Build the FAISS index first.')
    try:
        result = rag_chain.query(question=request.question, top_k=request.top_k)
    except Exception as exc:
        logger.error(f'RAG query failed: {exc}')
        raise HTTPException(status_code=500, detail=str(exc))
    scores = [c.get('score', 0.0) for c in result.get('source_documents', [])]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if avg_score >= 0.75:
        confidence = 'High'
    elif avg_score >= 0.5:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    sources = [f"{c.get('metadata', {}).get('document', 'Unknown')} — Page {c.get('metadata', {}).get('page', '?')}" for c in result.get('source_documents', [])]
    explainability = request.filters and request.filters.get('explainability') == 'true'
    retrieved_chunks = [{'text': c.get('page_content', ''), 'score': c.get('score', 0.0), 'metadata': c.get('metadata', {})} for c in result.get('source_documents', [])] if explainability else None
    return QueryResponse(question=request.question, answer=result.get('answer', 'The documents do not contain enough information.'), confidence=confidence, sources=list(dict.fromkeys(sources)), retrieved_chunks=retrieved_chunks)