from __future__ import annotations
import json
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models import VerifyRequest, VerifyResponse, VerificationResultOut
from app.dependencies import get_claims_df, get_documents, get_verification_pipeline
router = APIRouter(prefix='/verify', tags=['Verify'])
RESULTS_PATH = Path('data/processed/verification_results.json')

def _load_cached_results() -> List[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return []

def _map_result(idx: int, r: dict) -> VerificationResultOut:
    return VerificationResultOut(claim_id=idx, claim_text=r.get('claim_text', r.get('text', '')), esg_label=r.get('esg_label', 'MIXED'), risk_level=r.get('risk_level', r.get('aggregate_verdict', 'LOW')), l1_status=r.get('l1_status', 'UNKNOWN'), l2_status=r.get('l2_status', 'UNKNOWN'), l3_status=r.get('l3_status', 'UNKNOWN'), flags=r.get('flags', []), aggregate_verdict=r.get('aggregate_verdict', 'LOW'), confidence=float(r.get('confidence', 0.5)))

@router.get('', response_model=VerifyResponse)
async def get_verification_results() -> VerifyResponse:
    raw = _load_cached_results()
    if not raw:
        raise HTTPException(status_code=404, detail='No verification results found. POST /verify first.')
    results = [_map_result(i, r) for i, r in enumerate(raw)]
    high = sum((1 for r in results if r.risk_level.upper() == 'HIGH'))
    med = sum((1 for r in results if r.risk_level.upper() == 'MEDIUM'))
    low = sum((1 for r in results if r.risk_level.upper() == 'LOW'))
    return VerifyResponse(total_verified=len(results), high_risk=high, medium_risk=med, low_risk=low, results=results)

@router.post('', response_model=VerifyResponse)
async def run_verification(request: VerifyRequest) -> VerifyResponse:
    pipeline = get_verification_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail='VerificationPipeline failed to initialise.')
    df = get_claims_df()
    if df.empty:
        raise HTTPException(status_code=404, detail='No claims loaded. Run extraction first.')
    docs = get_documents()
    if not docs:
        raise HTTPException(status_code=404, detail='No documents loaded.')
    if request.claim_ids:
        df = df[df.index.isin(request.claim_ids)]
        if df.empty:
            raise HTTPException(status_code=404, detail='None of the requested claim IDs were found.')
    try:
        from src.extraction.claim_detector import Claim
        from src.extraction.pdf_extractor import ExtractedDocument
        claims = [Claim(text=row['text'], start_char=int(row.get('start_char', 0)), end_char=int(row.get('end_char', 0)), evidence_sentence=row.get('evidence_sentence', ''), evidence=row.get('evidence', ''), source_section=row.get('source_section', 'general'), claim_type=row.get('claim_type', 'quantitative'), confidence=float(row.get('confidence', 0.5)), esg_label=row.get('esg_label', 'MIXED')) for _, row in df.iterrows()]
        doc = ExtractedDocument(**docs[0]) if docs else None
        agg_results = pipeline.run(claims=claims, document=doc)
        raw_out = [r.to_dict() if hasattr(r, 'to_dict') else r.__dict__ for r in agg_results]
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(raw_out, f, indent=2, default=str)
        results = [_map_result(i, r) for i, r in enumerate(raw_out)]
        high = sum((1 for r in results if r.risk_level.upper() == 'HIGH'))
        med = sum((1 for r in results if r.risk_level.upper() == 'MEDIUM'))
        low = sum((1 for r in results if r.risk_level.upper() == 'LOW'))
        return VerifyResponse(total_verified=len(results), high_risk=high, medium_risk=med, low_risk=low, results=results)
    except Exception as exc:
        logger.error(f'Verification failed: {exc}')
        raise HTTPException(status_code=500, detail=str(exc))