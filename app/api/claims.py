"""
app/api/claims.py
==================
GET /claims        — list all extracted ESG claims (filterable)
GET /claims/{id}   — get a single claim by ID
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.models import ClaimsListResponse, ClaimOut
from app.dependencies import get_claims_df, get_finbert

router = APIRouter(prefix="/claims", tags=["Claims"])


def _row_to_claim(idx: int, row) -> ClaimOut:
    return ClaimOut(
        claim_id=idx,
        text=row.get("text", ""),
        evidence_sentence=row.get("evidence_sentence", ""),
        evidence=row.get("evidence", ""),
        source_section=row.get("source_section", "general"),
        claim_type=row.get("claim_type", "quantitative"),
        confidence=float(row.get("confidence", 0.0)),
        esg_label=row.get("esg_label", "MIXED"),
        document=row.get("document", ""),
        page=int(row["page"]) if row.get("page") is not None else None,
        matched_patterns=row.get("matched_patterns", []) if isinstance(row.get("matched_patterns"), list) else [],
    )


@router.get("", response_model=ClaimsListResponse)
async def list_claims(
    esg_label: Optional[str] = Query(None, description="Filter by E, S, G, or MIXED"),
    claim_type: Optional[str] = Query(None, description="Filter by quantitative/commitment/compliance"),
    document: Optional[str] = Query(None, description="Filter by document filename substring"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> ClaimsListResponse:
    """
    Return all extracted ESG claims with optional filters.
    Supports pagination via `limit` and `offset`.
    """
    df = get_claims_df()
    if df.empty:
        return ClaimsListResponse(total=0, claims=[])

    # Apply filters
    if esg_label:
        df = df[df["esg_label"].str.upper() == esg_label.upper()]
    if claim_type:
        df = df[df["claim_type"].str.lower() == claim_type.lower()]
    if document:
        df = df[df["document"].str.contains(document, case=False, na=False)]
    if min_confidence > 0.0:
        df = df[df["confidence"] >= min_confidence]

    total = len(df)
    page_df = df.iloc[offset : offset + limit]

    claims = [_row_to_claim(int(df.index[i]), row) for i, (_, row) in enumerate(page_df.iterrows())]
    return ClaimsListResponse(total=total, claims=claims)


@router.get("/{claim_id}", response_model=ClaimOut)
async def get_claim(
    claim_id: int,
    explain: bool = Query(False, description="If true, compute SHAP token attributions for this claim"),
) -> ClaimOut:
    """
    Return a single claim by its numeric ID.

    Pass `?explain=true` to include SHAP token-level attributions in the response.
    Note: SHAP computation takes ~3–8s on CPU and is only done for flagged claims.
    """
    df = get_claims_df()
    if df.empty or claim_id not in df.index:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    row = df.loc[claim_id]
    claim_out = _row_to_claim(claim_id, row)

    if explain:
        try:
            from src.classification.shap_explainer import ShapExplainer
            from app.models import ShapExplanationOut, ShapTokenOut

            finbert = get_finbert()
            explainer = ShapExplainer(classifier=finbert)
            shap_result = explainer.explain(
                claim_text=claim_out.text,
                prediction_label=claim_out.esg_label,
            )
            claim_out.shap_explanation = ShapExplanationOut(
                narrative=shap_result.narrative,
                top_risk_tokens=[
                    ShapTokenOut(token=t.token, value=t.value)
                    for t in shap_result.top_risk_tokens
                ],
                top_safe_tokens=[
                    ShapTokenOut(token=t.token, value=t.value)
                    for t in shap_result.top_safe_tokens
                ],
                prediction_label=shap_result.prediction_label,
                base_value=shap_result.base_value,
            )
        except Exception as exc:
            logger.warning(f"SHAP explanation failed for claim {claim_id}: {exc}")
            # Return claim without SHAP rather than 500

    return claim_out

