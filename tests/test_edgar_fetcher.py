"""
tests/test_edgar_fetcher.py
=============================
Unit tests for the EDGAR fetcher module.
HTTP calls are mocked to avoid hitting the live EDGAR API during CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion.edgar_fetcher import EdgarFetcher, FilingMetadata


# ── Sample EDGAR API responses ────────────────────────────────────────────────

SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}

SAMPLE_SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-K"],
            "filingDate": ["2023-11-03", "2023-08-04", "2022-10-28"],
            "accessionNumber": ["0000320193-23-000106", "0000320193-23-000077", "0000320193-22-000108"],
            "primaryDocument": ["aapl-20230930.htm", "aapl-20230701.htm", "aapl-20220924.htm"],
            "primaryDocDescription": ["10-K", "10-Q", "10-K"],
            "reportDate": ["2023-09-30", "2023-07-01", "2022-09-24"],
        }
    }
}


class TestEdgarFetcher:

    @pytest.fixture
    def fetcher(self):
        return EdgarFetcher(user_agent="AuditLens test@test.com")

    def _mock_response(self, data: dict, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    # ── CIK resolution ────────────────────────────────────────────────────────

    @patch("requests.Session.get")
    def test_resolve_cik_known_ticker(self, mock_get, fetcher):
        mock_get.return_value = self._mock_response(SAMPLE_COMPANY_TICKERS)
        cik = fetcher._resolve_cik("AAPL")
        assert cik == "0000320193"

    @patch("requests.Session.get")
    def test_resolve_cik_case_insensitive(self, mock_get, fetcher):
        mock_get.return_value = self._mock_response(SAMPLE_COMPANY_TICKERS)
        cik = fetcher._resolve_cik("aapl")
        assert cik == "0000320193"

    @patch("requests.Session.get")
    def test_resolve_cik_unknown_ticker(self, mock_get, fetcher):
        mock_get.return_value = self._mock_response(SAMPLE_COMPANY_TICKERS)
        cik = fetcher._resolve_cik("XYZNOTREAL")
        assert cik is None

    @patch("requests.Session.get")
    def test_resolve_cik_network_error(self, mock_get, fetcher):
        mock_get.side_effect = requests.RequestException("Network error")
        cik = fetcher._resolve_cik("AAPL")
        assert cik is None

    # ── Filing fetch ──────────────────────────────────────────────────────────

    @patch("requests.Session.get")
    def test_fetch_recent_filings_returns_10k(self, mock_get, fetcher):
        mock_get.side_effect = [
            self._mock_response(SAMPLE_COMPANY_TICKERS),  # CIK resolution
            self._mock_response(SAMPLE_SUBMISSIONS),       # submissions
        ]
        # Patch cutoff date to be far in the past
        with patch("src.ingestion.edgar_fetcher.datetime") as mock_dt:
            from datetime import datetime, timedelta
            mock_dt.utcnow.return_value = datetime(2025, 1, 1)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            filings = fetcher.fetch_recent_filings("AAPL", form_type="10-K", days_back=9999)

        assert len(filings) == 2
        assert all(f.form_type == "10-K" for f in filings)
        assert filings[0].filing_date >= filings[1].filing_date  # newest first

    @patch("requests.Session.get")
    def test_get_historical_filing(self, mock_get, fetcher):
        mock_get.side_effect = [
            self._mock_response(SAMPLE_COMPANY_TICKERS),
            self._mock_response(SAMPLE_SUBMISSIONS),
        ]
        filing = fetcher.get_historical_filing("AAPL", year=2022)
        assert filing is not None
        assert "2022" in filing.period_of_report

    @patch("requests.Session.get")
    def test_filing_metadata_structure(self, mock_get, fetcher):
        mock_get.side_effect = [
            self._mock_response(SAMPLE_COMPANY_TICKERS),
            self._mock_response(SAMPLE_SUBMISSIONS),
        ]
        with patch("src.ingestion.edgar_fetcher.datetime") as mock_dt:
            from datetime import datetime
            mock_dt.utcnow.return_value = datetime(2025, 1, 1)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            filings = fetcher.fetch_recent_filings("AAPL", form_type="10-K", days_back=9999)

        if filings:
            f = filings[0]
            assert isinstance(f, FilingMetadata)
            assert f.company_name == "Apple Inc."
            assert f.cik is not None
            assert f.document_url.startswith("https://")

    # ── Download ──────────────────────────────────────────────────────────────

    @patch("requests.Session.get")
    def test_download_filing_creates_file(self, mock_get, fetcher, tmp_path):
        # Mock the download response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"fake pdf content"]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        filing = FilingMetadata(
            cik="0000320193",
            accession_number="0000320193-23-000106",
            company_name="Apple Inc.",
            form_type="10-K",
            filing_date="2023-11-03",
            primary_document="test.pdf",
            document_url="https://www.sec.gov/test.pdf",
        )

        result = fetcher.download_filing(filing, output_dir=tmp_path)
        assert result.download_success
        assert result.local_path.exists()

    @patch("requests.Session.get")
    def test_download_skips_if_cached(self, mock_get, fetcher, tmp_path):
        filing = FilingMetadata(
            cik="0000320193",
            accession_number="0000320193-23-000106",
            company_name="Apple Inc.",
            form_type="10-K",
            filing_date="2023-11-03",
            primary_document="test.pdf",
            document_url="https://www.sec.gov/test.pdf",
        )
        # Create the file so it appears cached
        cached = tmp_path / "000032019323000106_test.pdf"
        cached.write_bytes(b"cached content")

        result = fetcher.download_filing(filing, output_dir=tmp_path)
        # Should not have made any HTTP calls
        mock_get.assert_not_called()
        assert result.local_path.exists()
