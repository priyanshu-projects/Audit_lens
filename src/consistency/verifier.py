"""
src/consistency/verifier.py
============================
Orchestrates L1, L2, and L3 verification checks for a list of claims
against an ExtractedDocument.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

from src.extraction.claim_detector import Claim
from src.extraction.pdf_extractor import ExtractedDocument
from src.consistency.internal_checker import InternalChecker, ConsistencyResult
from src.consistency.historical_checker import HistoricalChecker
from src.consistency.standard_checker import StandardChecker
from src.consistency.aggregator import aggregate_results, AggregateResult
from src.rag.vector_store import VectorStore
from config.settings import app_cfg


class VerificationPipeline:
    """
    Unified pipeline to execute all level checks (L1 Internal, L2 Historical, L3 Standards Compliance)
    on claims and aggregate the final verdicts.
    """

    def __init__(
        self,
        l1_checker: Optional[InternalChecker] = None,
        l2_checker: Optional[HistoricalChecker] = None,
        l3_checker: Optional[StandardChecker] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.l1_checker = l1_checker or InternalChecker()
        self.l2_checker = l2_checker or HistoricalChecker()
        
        # Load vector store if not provided for Level 3 Standard checker
        if not l3_checker:
            vs = vector_store
            if not vs:
                try:
                    index_path = app_cfg.faiss_index_path
                    if index_path.exists() and (index_path / "faiss.index").exists():
                        vs = VectorStore.load(index_path)
                except Exception:
                    vs = None
            
            if vs:
                self.l3_checker = StandardChecker(vs)
            else:
                self.l3_checker = None
        else:
            self.l3_checker = l3_checker

    def verify_claim(self, claim: Claim, doc: ExtractedDocument) -> dict:
        """
        Run all verification levels on a single claim.

        Returns:
            dict containing:
                "claim": the original Claim object
                "l1": ConsistencyResult for L1
                "l2": ConsistencyResult for L2
                "l3": ConsistencyResult for L3
                "agg_result": final aggregated AggregateResult object
        """
        # Level 1: Internal document consistency
        l1 = self.l1_checker.check(claim.text, claim.start_char, doc)

        # Level 2: Historical cross-check
        try:
            filing_year_int = int(doc.filing_year) if doc.filing_year else 0
        except (ValueError, TypeError):
            filing_year_int = 0
        l2 = self.l2_checker.check(claim, doc.ticker, filing_year_int)

        # Level 3: Standards check (GRI/TCFD/ISSB/SASB)
        if self.l3_checker:
            l3 = self.l3_checker.check(claim)
        else:
            # Fallback if FAISS index is not built
            l3 = ConsistencyResult(
                level=3,
                status="SKIPPED",
                note="FAISS vector store index not loaded — L3 standards verification skipped.",
            )

        # Aggregate L1–L3 checks into a single verdict
        agg = aggregate_results(
            l1=l1,
            l2=l2,
            l3=l3,
            esg_label=claim.esg_label,
            nlp_confidence=claim.confidence,
        )

        return {
            "claim": claim,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "agg_result": agg,
        }

    def verify_batch(self, claims: list[Claim], doc: ExtractedDocument) -> list[dict]:
        """Verify a batch of claims against the document context."""
        results = []
        for claim in claims:
            res = self.verify_claim(claim, doc)
            results.append(res)
        return results
