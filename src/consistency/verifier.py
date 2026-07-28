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

    def __init__(self, l1_checker: Optional[InternalChecker]=None, l2_checker: Optional[HistoricalChecker]=None, l3_checker: Optional[StandardChecker]=None, vector_store: Optional[VectorStore]=None) -> None:
        self.l1_checker = l1_checker or InternalChecker()
        self.l2_checker = l2_checker or HistoricalChecker()
        if not l3_checker:
            vs = vector_store
            if not vs:
                try:
                    index_path = app_cfg.faiss_index_path
                    if index_path.exists() and (index_path / 'faiss.index').exists():
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
        l1 = self.l1_checker.check(claim.text, claim.start_char, doc)
        try:
            filing_year_int = int(doc.filing_year) if doc.filing_year else 0
        except (ValueError, TypeError):
            filing_year_int = 0
        l2 = self.l2_checker.check(claim, doc.ticker, filing_year_int)
        if self.l3_checker:
            l3 = self.l3_checker.check(claim)
        else:
            l3 = ConsistencyResult(level=3, status='SKIPPED', note='FAISS vector store index not loaded — L3 standards verification skipped.')
        agg = aggregate_results(l1=l1, l2=l2, l3=l3, esg_label=claim.esg_label, nlp_confidence=claim.confidence)
        return {'claim': claim, 'l1': l1, 'l2': l2, 'l3': l3, 'agg_result': agg}

    def verify_batch(self, claims: list[Claim], doc: ExtractedDocument) -> list[dict]:
        results = []
        for claim in claims:
            res = self.verify_claim(claim, doc)
            results.append(res)
        return results

    def run(self, claims: list[Claim], document: ExtractedDocument) -> list[PipelineVerificationResult]:
        results = self.verify_batch(claims, document)
        flat_results = []
        for i, item in enumerate(results):
            claim = item['claim']
            l1 = item['l1']
            l2 = item['l2']
            l3 = item['l3']
            agg = item['agg_result']
            flat_results.append(PipelineVerificationResult({'claim_id': i + 1, 'claim_text': claim.text, 'esg_label': claim.esg_label, 'risk_level': 'HIGH' if agg.verdict in ['HIGH_RISK', 'INCONSISTENT'] else 'MEDIUM' if agg.verdict == 'PARTIALLY_CONSISTENT' else 'LOW', 'l1_status': l1.status, 'l1_note': l1.note, 'l2_status': l2.status, 'l2_note': l2.note, 'l3_status': l3.status, 'l3_note': l3.note, 'flags': [f'L1_{l1.status}', f'L2_{l2.status}', f'L3_{l3.status}'] if agg.verdict != 'CONSISTENT' else [], 'aggregate_verdict': agg.verdict, 'confidence': float(claim.confidence), 'risk_score': float(agg.risk_score), 'final_note': agg.final_note}))
        return flat_results

class PipelineVerificationResult:

    def __init__(self, data: dict) -> None:
        self.__dict__.update(data)

    def to_dict(self) -> dict:
        return self.__dict__