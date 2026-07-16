"""
app/models.py
  — Added ShapTokenOut + ShapExplanationOut for SHAP API responses
==============
Pydantic request/response models for the AuditLens FastAPI.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── SHAP ──────────────────────────────────────────────────────────────────────

class ShapTokenOut(BaseModel):
    """SHAP attribution for a single token."""
    token: str
    value: float   # positive = risk-raising, negative = risk-lowering


class ShapExplanationOut(BaseModel):
    """Full SHAP explanation for one claim — returned by GET /claims/{id}?explain=true."""
    narrative: str
    top_risk_tokens: List[ShapTokenOut]   # words that raised risk score
    top_safe_tokens: List[ShapTokenOut]   # words that lowered risk score
    prediction_label: str
    base_value: float


# ── Shared ────────────────────────────────────────────────────────────────────

class ClaimOut(BaseModel):
    """A single extracted ESG claim returned from the API."""
    claim_id: int
    text: str
    evidence_sentence: str
    evidence: str
    source_section: str
    claim_type: str
    confidence: float
    esg_label: str
    document: str
    page: Optional[int] = None
    matched_patterns: List[str] = Field(default_factory=list)
    shap_explanation: Optional[ShapExplanationOut] = None


class VerificationResultOut(BaseModel):
    """Verification result for a single claim."""
    claim_id: int
    claim_text: str
    esg_label: str
    risk_level: str                  # HIGH / MEDIUM / LOW
    l1_status: str
    l2_status: str
    l3_status: str
    flags: List[str] = Field(default_factory=list)
    aggregate_verdict: str
    confidence: float


class AuditObservation(BaseModel):
    """One observation in the final audit report."""
    claim_id: int
    claim_text: str
    risk_level: str
    observation: str
    recommendation: str
    sources: List[str] = Field(default_factory=list)


# ── /query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., example="What emission reduction targets has Apple committed to?")
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, str]] = Field(
        None,
        example={"esg_label": "E", "document": "aapl-20240928"},
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str          # High / Medium / Low
    sources: List[str]
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None   # explainability mode


# ── /claims ────────────────────────────────────────────────────────────────────

class ClaimsListResponse(BaseModel):
    total: int
    claims: List[ClaimOut]


# ── /verify ────────────────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """Trigger on-demand verification for specific claim IDs (or all)."""
    claim_ids: Optional[List[int]] = Field(
        None,
        description="If omitted, verifies all loaded claims.",
    )


class VerifyResponse(BaseModel):
    total_verified: int
    high_risk: int
    medium_risk: int
    low_risk: int
    results: List[VerificationResultOut]


# ── /report ────────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    format: str = Field("json", pattern="^(json|markdown)$")


class ReportResponse(BaseModel):
    generated_at: str
    total_claims: int
    high_risk_count: int
    observations: List[AuditObservation]
    markdown_summary: Optional[str] = None


# ── /documents ─────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    company: str
    year: int
    total_pages: int
    total_claims: int
    esg_breakdown: Dict[str, int]


class DocumentsListResponse(BaseModel):
    total: int
    documents: List[DocumentOut]


# ── /upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    status: str          # "queued" | "processing" | "done" | "error"
    message: str


# ── /health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    claims_loaded: int
    index_loaded: bool
    model: str
