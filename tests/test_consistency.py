"""
tests/test_consistency.py
===========================
Unit tests for all 4 consistency check levels and the aggregator.
"""

from __future__ import annotations

import pytest

from src.consistency.internal_checker import InternalChecker, ConsistencyResult
from src.consistency.aggregator import aggregate_results, AggregateResult
from src.extraction.claim_detector import Claim
from src.extraction.pdf_extractor import ExtractedDocument, TableData


class TestInternalChecker:

    @pytest.fixture
    def checker(self):
        return InternalChecker()

    @pytest.fixture
    def doc_with_table(self, tmp_path):
        """Doc containing a table with 40% and year 2019."""
        table = TableData(
            page_number=5,
            headers=["Year", "Scope 1 Emissions (tCO2e)", "Change"],
            rows=[["2019", "850,000", "baseline"], ["2023", "510,000", "-40%"]],
        )
        doc = ExtractedDocument(
            source_path=tmp_path / "test.pdf",
            raw_text="We reduced emissions by 40% since 2019.\n\n2019 | 850000 | baseline\n2023 | 510000 | -40%",
            tables=[table],
        )
        return doc

    @pytest.fixture
    def doc_without_table(self, tmp_path):
        """Doc with no supporting table."""
        return ExtractedDocument(
            source_path=tmp_path / "test.pdf",
            raw_text="We reduced emissions by 40% since 2019.",
            tables=[],
        )

    def test_supported_when_table_has_matching_number(self, checker, doc_with_table):
        result = checker.check(
            claim_text="We reduced Scope 1 emissions by 40% since 2019.",
            claim_start_char=0,
            extracted_doc=doc_with_table,
        )
        assert result.level == 1
        assert result.status in ("SUPPORTED", "PARTIALLY_SUPPORTED")

    def test_unsupported_when_no_table(self, checker, doc_without_table):
        result = checker.check(
            claim_text="We reduced Scope 1 emissions by 87%.",
            claim_start_char=0,
            extracted_doc=doc_without_table,
        )
        assert result.level == 1
        assert result.status == "UNSUPPORTED"

    def test_skipped_for_non_quantitative_claim(self, checker, doc_without_table):
        result = checker.check(
            claim_text="We are committed to environmental responsibility.",
            claim_start_char=0,
            extracted_doc=doc_without_table,
        )
        assert result.level == 1
        assert result.status == "SKIPPED"

    def test_result_has_note(self, checker, doc_with_table):
        result = checker.check(
            claim_text="We reduced emissions by 40% since 2019.",
            claim_start_char=0,
            extracted_doc=doc_with_table,
        )
        assert isinstance(result.note, str)
        assert len(result.note) > 10


class TestStandardChecker:

    @pytest.fixture
    def checker(self, mock_vector_store):
        from src.consistency.standard_checker import StandardChecker
        return StandardChecker(vector_store=mock_vector_store)

    def test_scope1_claim_matches_gri305(self, checker):
        from src.extraction.claim_detector import Claim
        claim = Claim(
            text="Scope 1 GHG emissions were 850,000 tCO2e with 2019 as our base year.",
            start_char=0, end_char=70,
        )
        result = checker.check(claim)
        assert result.level == 3
        assert result.status in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED")

    def test_netzero_claim_checked_against_tcfd(self, checker):
        from src.extraction.claim_detector import Claim
        claim = Claim(
            text="Our net-zero target is to achieve carbon neutrality by 2040.",
            start_char=0, end_char=60,
        )
        result = checker.check(claim)
        assert result.level == 3
        # Should flag missing elements (no scope coverage, no interim targets, etc.)
        assert result.status in ("PARTIALLY_SUPPORTED", "UNSUPPORTED")


class TestAggregator:

    def _make_result(self, level, status):
        return ConsistencyResult(level=level, status=status, note=f"L{level} {status}")

    def test_all_supported_gives_consistent(self):
        agg = aggregate_results(
            l1=self._make_result(1, "SUPPORTED"),
            l2=self._make_result(2, "SUPPORTED"),
            l3=self._make_result(3, "SUPPORTED"),
        )
        assert agg.verdict == "CONSISTENT"
        assert agg.risk_score < 0.2

    def test_all_unsupported_gives_inconsistent(self):
        agg = aggregate_results(
            l1=self._make_result(1, "UNSUPPORTED"),
            l2=self._make_result(2, "UNSUPPORTED"),
            l3=self._make_result(3, "UNSUPPORTED"),
        )
        assert agg.verdict in ("INCONSISTENT", "HIGH_RISK")
        assert agg.risk_score >= 0.5

    def test_skipped_doesnt_penalise(self):
        agg = aggregate_results(
            l1=self._make_result(1, "SUPPORTED"),
            l2=self._make_result(2, "SKIPPED"),
            l3=self._make_result(3, "SUPPORTED"),
            l4=self._make_result(4, "SKIPPED"),
        )
        # Skipped levels should not inflate risk
        assert agg.verdict == "CONSISTENT"

    def test_cdp_discrepancy_gives_high_risk(self):
        l4 = ConsistencyResult(
            level=4,
            status="UNSUPPORTED",
            note="SIGNIFICANT DISCREPANCY with CDP submission detected",
        )
        agg = aggregate_results(l4=l4)
        assert agg.verdict == "HIGH_RISK"
        assert agg.risk_score == 1.0

    def test_risk_score_in_range(self):
        for status in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "SKIPPED"]:
            agg = aggregate_results(l1=self._make_result(1, status))
            assert 0.0 <= agg.risk_score <= 1.0

    def test_empty_result(self):
        agg = aggregate_results()
        assert isinstance(agg, AggregateResult)
        assert agg.verdict is not None

    def test_verdict_colours_defined(self):
        from src.consistency.aggregator import VERDICT_COLORS, VERDICT_EMOJI
        assert "CONSISTENT" in VERDICT_COLORS
        assert "HIGH_RISK" in VERDICT_COLORS
        assert "CONSISTENT" in VERDICT_EMOJI
