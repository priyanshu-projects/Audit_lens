"""
src/consistency/aggregator.py
================================
Aggregates L1–L3 consistency check results into a final audit verdict.

Final verdicts:
  CONSISTENT        — All available checks pass, low risk
  PARTIALLY_CONSISTENT — Some checks pass, some fail; medium risk
  INCONSISTENT      — Multiple checks fail; high risk
  HIGH_RISK         — At least one L3 SBTi discrepancy OR critical flag

The aggregator also computes a numeric risk score (0.0–1.0) that is used
by the Streamlit heatmap to colour-code claims by section (E/S/G).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.consistency.internal_checker import ConsistencyResult


# ── Status scoring weights ────────────────────────────────────────────────────
# Maps status strings to a risk contribution (0.0 = no risk, 1.0 = max risk)

STATUS_SCORES = {
    "SUPPORTED": 0.0,
    "PARTIALLY_SUPPORTED": 0.4,
    "UNSUPPORTED": 0.8,
    "SKIPPED": 0.0,     # SKIPPED doesn't penalise — data just wasn't available
    "HIGH_RISK": 1.0,
}

# Level weights — L3 (SBTi External Verification) is most authoritative
LEVEL_WEIGHTS = {
    1: 0.25,   # Internal document consistency
    2: 0.35,   # Historical cross-check
    3: 0.40,   # SBTi External Verification
    4: 0.50,   # CDP verification check
}


@dataclass
class AggregateResult:
    """Final consistency verdict across all available check levels."""
    verdict: str                     # CONSISTENT | PARTIALLY_CONSISTENT | INCONSISTENT | HIGH_RISK
    risk_score: float                # 0.0–1.0
    summary: str                     # one-line summary for dashboard
    level_results: dict[int, ConsistencyResult] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)   # specific issues to highlight

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "risk_score": round(self.risk_score, 4),
            "summary": self.summary,
            "flags": self.flags,
            "levels": {
                lvl: {
                    "status": r.status,
                    "note": r.note,
                }
                for lvl, r in self.level_results.items()
            },
        }


# ── Verdict colour mapping for Streamlit ─────────────────────────────────────

VERDICT_COLORS = {
    "CONSISTENT": "#22c55e",           # green
    "PARTIALLY_CONSISTENT": "#f59e0b", # amber
    "INCONSISTENT": "#ef4444",         # red
    "HIGH_RISK": "#7c3aed",            # purple
}

VERDICT_EMOJI = {
    "CONSISTENT": "✅",
    "PARTIALLY_CONSISTENT": "⚠️",
    "INCONSISTENT": "❌",
    "HIGH_RISK": "🚨",
}


# FinBERT flag → fallback risk when all checks are SKIPPED
# Note: if ESG model labels as MIXED ("None" in yiyanghkust/finbert-esg), the
# claim is not an ESG claim — use NOT_ESG_CLAIM verdict instead of INCONSISTENT.
_NLP_FLAG_FALLBACK: dict[str, tuple[float, str]] = {
    "HIGH_RISK":         (0.75, "INCONSISTENT"),
    "NEEDS_REVIEW":      (0.40, "PARTIALLY_CONSISTENT"),
    "LIKELY_CONSISTENT": (0.15, "CONSISTENT"),
}


def aggregate_results(
    l1: Optional[ConsistencyResult] = None,
    l2: Optional[ConsistencyResult] = None,
    l3: Optional[ConsistencyResult] = None,
    l4: Optional[ConsistencyResult] = None,
    nlp_flag: str = "",
    esg_label: str = "MIXED",
    nlp_confidence: float = 0.0,
) -> AggregateResult:
    """
    Aggregate consistency check results from all available levels.

    Args:
        l1:             Internal document consistency result (Level 1)
        l2:             Historical cross-check result (Level 2)
        l3:             SBTi external verification result (Level 3)
        l4:             CDP verification result (Level 4)
        nlp_flag:       FinBERT consistency_flag used as fallback when all
                        structural checks are SKIPPED.
        esg_label:      ESG label from FinBERT (E/S/G/MIXED). If MIXED with
                        high confidence, the claim is not an ESG claim.
        nlp_confidence: FinBERT confidence score for the top label.

    Returns:
        AggregateResult with final verdict and risk score
    """
    level_results: dict[int, ConsistencyResult] = {}
    flags: list[str] = []

    level_map = {1: l1, 2: l2, 3: l3, 4: l4}
    for lvl, result in level_map.items():
        if result is not None:
            level_results[lvl] = result

    # If ESG model labels as MIXED with high confidence, the sentence is
    # NOT an ESG claim — generic boilerplate that slipped through detection.
    # Short-circuit here to prevent random numbers from failing L1/L2 checks.
    if esg_label == "MIXED" and nlp_confidence >= 0.80:
        return AggregateResult(
            verdict="NOT_ESG_CLAIM",
            risk_score=0.05,
            summary=(
                "FinBERT ESG model determined this sentence is not an ESG claim "
                f"(confidence: {nlp_confidence:.0%}). Likely corporate boilerplate."
            ),
            level_results=level_results,
            flags=["Not classified as E/S/G by fine-tuned ESG model."],
        )

    if not level_results:
        return AggregateResult(
            verdict="PARTIALLY_CONSISTENT",
            risk_score=0.5,
            summary="No consistency checks were run.",
            level_results={},
        )

    # Compute weighted risk score
    total_weight = 0.0
    weighted_score = 0.0

    for lvl, result in level_results.items():
        if result.status == "SKIPPED":
            continue
        weight = LEVEL_WEIGHTS.get(lvl, 0.2)
        score = STATUS_SCORES.get(result.status, 0.5)
        weighted_score += weight * score
        total_weight += weight
        if result.status == "UNSUPPORTED":
            flags.append(f"L{lvl}: {result.note[:100]}")

    # Check for SBTi discrepancy (immediate HIGH_RISK)
    if l3 and l3.status == "HIGH_RISK":
        return AggregateResult(
            verdict="HIGH_RISK",
            risk_score=1.0,
            summary="External verification discrepancy detected (SBTi conflict).",
            level_results=level_results,
            flags=flags + [f"L3 SBTi: {l3.note[:150]}"],
        )

    # Check for CDP discrepancy (immediate HIGH_RISK)
    if l4 and l4.status == "UNSUPPORTED":
        return AggregateResult(
            verdict="HIGH_RISK",
            risk_score=1.0,
            summary="CDP submission discrepancy detected.",
            level_results=level_results,
            flags=flags + [f"L4 CDP: {l4.note[:150]}"],
        )

    # ── All checks were SKIPPED (e.g. claim has no numeric content) ──────────
    # Structural scoring cannot run. Fall back to FinBERT's NLP risk signal.
    if total_weight == 0.0:
        fallback_risk, fallback_verdict = _NLP_FLAG_FALLBACK.get(
            nlp_flag, (0.30, "PARTIALLY_CONSISTENT")
        )
        return AggregateResult(
            verdict=fallback_verdict,
            risk_score=round(fallback_risk, 4),
            summary=(
                "Structural consistency checks could not run for this claim "
                "(no numeric evidence to verify). Risk estimate is based on "
                f"FinBERT NLP analysis: {nlp_flag or 'unknown'}."
            ),
            level_results=level_results,
            flags=flags,
        )

    risk_score = weighted_score / max(total_weight, 0.01)

    # Determine verdict
    if risk_score < 0.2:
        verdict = "CONSISTENT"
        summary = "Claim is well-supported across all available evidence sources."
    elif risk_score < 0.5:
        verdict = "PARTIALLY_CONSISTENT"
        summary = "Claim has partial support — some disclosure gaps identified."
    elif risk_score < 0.8:
        verdict = "INCONSISTENT"
        summary = "Claim has significant inconsistencies — auditor review required."
    else:
        verdict = "HIGH_RISK"
        summary = "Multiple consistency failures — potential greenwashing risk."

    return AggregateResult(
        verdict=verdict,
        risk_score=round(risk_score, 4),
        summary=summary,
        level_results=level_results,
        flags=flags,
    )
