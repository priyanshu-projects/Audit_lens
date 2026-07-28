"""
app/api/report.py
==================
GET  /report       — return cached audit report (json or markdown)
POST /report       — regenerate the audit report on demand
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from loguru import logger

from app.models import ReportRequest, ReportResponse, AuditObservation
from app.dependencies import get_claims_df, get_documents

router = APIRouter(prefix="/report", tags=["Report"])

REPORT_JSON_PATH = Path("data/processed/audit_report.json")
REPORT_MD_PATH   = Path("data/processed/audit_report.md")


def _load_cached_report() -> dict:
    if REPORT_JSON_PATH.exists():
        with open(REPORT_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@router.get("", response_model=ReportResponse)
async def get_report(format: str = Query("json", pattern="^(json|markdown)$")) -> ReportResponse:
    """Return cached audit report from the last pipeline run."""
    if format == "markdown":
        if REPORT_MD_PATH.exists():
            return PlainTextResponse(REPORT_MD_PATH.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Markdown report not found.")

    data = _load_cached_report()
    if not data:
        raise HTTPException(
            status_code=404,
            detail="No audit report found. Run POST /report first.",
        )

    observations = [
        AuditObservation(
            claim_id=o.get("claim_id", i),
            claim_text=o.get("claim_text", ""),
            risk_level=o.get("risk_level", "LOW"),
            observation=o.get("observation", ""),
            recommendation=o.get("recommendation", ""),
            sources=o.get("sources", []),
        )
        for i, o in enumerate(data.get("observations", []))
    ]

    return ReportResponse(
        generated_at=data.get("generated_at", ""),
        total_claims=data.get("total_claims", 0),
        high_risk_count=data.get("high_risk_count", 0),
        observations=observations,
        markdown_summary=data.get("markdown_summary"),
    )


@router.post("", response_model=ReportResponse)
async def regenerate_report(request: ReportRequest) -> ReportResponse:
    """Regenerate the audit report on demand from current verification results."""
    ver_path = Path("data/processed/verification_results.json")
    if not ver_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Verification results not found. Run POST /verify first.",
        )

    try:
        from src.reporting.report_generator import ReportGenerator
        generator = ReportGenerator()

        with open(ver_path, encoding="utf-8") as f:
            ver_results = json.load(f)

        report = generator.generate(
            verification_results=ver_results,
            output_json_path=REPORT_JSON_PATH,
            output_md_path=REPORT_MD_PATH,
        )

        # Persist
        REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(report if isinstance(report, dict) else report.__dict__, f, indent=2, default=str)

        if request.format == "markdown" and REPORT_MD_PATH.exists():
            return PlainTextResponse(REPORT_MD_PATH.read_text(encoding="utf-8"))

        data = _load_cached_report()
        observations = [
            AuditObservation(
                claim_id=o.get("claim_id", i),
                claim_text=o.get("claim_text", ""),
                risk_level=o.get("risk_level", "LOW"),
                observation=o.get("observation", ""),
                recommendation=o.get("recommendation", ""),
                sources=o.get("sources", []),
            )
            for i, o in enumerate(data.get("observations", []))
        ]
        return ReportResponse(
            generated_at=data.get("generated_at", datetime.now(timezone.utc).isoformat()),
            total_claims=data.get("total_claims", 0),
            high_risk_count=data.get("high_risk_count", 0),
            observations=observations,
        )

    except Exception as exc:
        logger.error(f"Report generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
