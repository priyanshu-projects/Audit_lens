"""
src/ingestion/edgar_fetcher.py
===============================
Fetches ESG-related filings from the SEC EDGAR REST API.

EDGAR API is completely public — no API key required.
EDGAR requires a descriptive User-Agent header (see .env.example).

Key endpoints used:
  - https://data.sec.gov/submissions/CIK{cik}.json        → company filing history
  - https://efts.sec.gov/LATEST/search-index?q=...        → full-text search
  - https://www.sec.gov/Archives/edgar/full-index/          → direct filing index



Usage:
    fetcher = EdgarFetcher()
    filings = fetcher.fetch_recent_filings("AAPL", form_type="10-K")
    for filing in filings:
        path = fetcher.download_filing(filing, output_dir=Path("data/raw"))
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from loguru import logger

from config.settings import edgar_cfg


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""
    cik: str
    accession_number: str
    company_name: str
    form_type: str
    filing_date: str          # ISO format: "2023-11-15"
    primary_document: str     # filename of the main document
    document_url: str         # full URL to the primary document
    description: str = ""
    period_of_report: str = ""


@dataclass
class DownloadedFiling:
    """Result of downloading a filing to disk."""
    metadata: FilingMetadata
    local_path: Path
    raw_text: str = ""        # populated by pdf_extractor later
    download_success: bool = True
    error: str = ""


# ── Main class ───────────────────────────────────────────────────────────────

class EdgarFetcher:
    """
    Fetches ESG filings from SEC EDGAR.

    Implements:
    - CIK resolution from ticker symbol
    - Recent filing lookup (daily ingestion job)
    - Historical filing lookup (for L2 consistency checks)
    - PDF/HTML downloading with retry logic
    """

    BASE_URL = "https://data.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar"

    # EDGAR throttles aggressively — max 10 requests/second
    REQUEST_DELAY_SECONDS = 0.12

    # Form types we consider as ESG-related filings
    ESG_FORM_TYPES = ["10-K", "10-K/A", "20-F", "DEF 14A", "8-K"]

    def __init__(self, user_agent: str = edgar_cfg.user_agent):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        })
        self._last_request_time: float = 0.0
        logger.info(f"EdgarFetcher initialised | user_agent='{user_agent}'")

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_recent_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        days_back: int = edgar_cfg.lookback_days,
    ) -> list[FilingMetadata]:
        """
        Return filings for `ticker` of `form_type` filed in the last `days_back` days.

        Args:
            ticker:    Stock ticker, e.g. "AAPL", "MSFT"
            form_type: SEC form type, e.g. "10-K" for annual reports
            days_back: How far back to look

        Returns:
            List of FilingMetadata objects, newest first
        """
        logger.info(f"Fetching recent {form_type} filings for {ticker} (last {days_back}d)")
        cik = self._resolve_cik(ticker)
        if not cik:
            logger.error(f"Could not resolve CIK for ticker '{ticker}'")
            return []

        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        all_filings = self._get_filings_for_cik(cik, form_type)

        recent = [f for f in all_filings if f.filing_date >= cutoff_date]
        logger.info(f"Found {len(recent)} recent {form_type} filings for {ticker}")
        return recent

    def get_historical_filing(
        self,
        ticker: str,
        year: int,
        form_type: str = "10-K",
    ) -> Optional[FilingMetadata]:
        """
        Return the filing closest to the end of `year` for L2 consistency checks.

        Args:
            ticker:    Stock ticker
            year:      Calendar year, e.g. 2022
            form_type: Usually "10-K"

        Returns:
            FilingMetadata for the filing covering that year, or None
        """
        logger.info(f"Fetching historical {form_type} for {ticker} year={year}")
        cik = self._resolve_cik(ticker)
        if not cik:
            return None

        all_filings = self._get_filings_for_cik(cik, form_type)

        # Find the filing whose period_of_report falls in that year
        for filing in all_filings:
            if filing.period_of_report.startswith(str(year)):
                logger.info(f"Found historical filing: {filing.accession_number}")
                return filing

        logger.warning(f"No {form_type} found for {ticker} year {year}")
        return None

    def download_filing(
        self,
        filing: FilingMetadata,
        output_dir: Path,
    ) -> DownloadedFiling:
        """
        Download the primary document of a filing to disk.

        Handles both PDF and HTML/HTM filing types.
        Returns a DownloadedFiling with local_path set.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filing.accession_number.replace("-", "") + "_" + filing.primary_document
        local_path = output_dir / safe_name

        if local_path.exists():
            logger.debug(f"Filing already cached: {local_path}")
            return DownloadedFiling(metadata=filing, local_path=local_path)

        logger.info(f"Downloading filing {filing.accession_number} → {local_path}")
        try:
            resp = self._get_with_retry(filing.document_url, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.success(f"Downloaded {local_path.name} ({local_path.stat().st_size:,} bytes)")
            return DownloadedFiling(metadata=filing, local_path=local_path)

        except Exception as exc:
            logger.error(f"Failed to download {filing.document_url}: {exc}")
            return DownloadedFiling(
                metadata=filing,
                local_path=local_path,
                download_success=False,
                error=str(exc),
            )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """
        Convert a stock ticker to a zero-padded 10-digit SEC CIK.
        Uses the EDGAR company tickers JSON (updated nightly by SEC).
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            resp = self._get_with_retry(url)
            resp.raise_for_status()
            tickers_data = resp.json()
            ticker_upper = ticker.upper()
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.debug(f"Resolved {ticker} → CIK {cik}")
                    return cik
            logger.warning(f"Ticker '{ticker}' not found in EDGAR company tickers")
            return None
        except Exception as exc:
            logger.error(f"CIK resolution failed for {ticker}: {exc}")
            return None

    def _get_filings_for_cik(
        self,
        cik: str,
        form_type: str,
    ) -> list[FilingMetadata]:
        """
        Fetch full filing history for a CIK and filter by form_type.
        Returns list sorted newest-first.
        """
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        try:
            resp = self._get_with_retry(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {exc}")
            return []

        company_name = data.get("name", "Unknown")
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        periods = recent.get("reportDate", [])

        results: list[FilingMetadata] = []
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            accession = accessions[i].replace("-", "")
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            doc_url = (
                f"{self.ARCHIVES_URL}/full-index/"
                f"{accession[:4]}/{accession[4:6]}/{accession[6:8]}/"
                f"{accession}/{primary_doc}"
            )
            # Build proper document URL
            acc_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession}/{primary_doc}"
            )
            results.append(FilingMetadata(
                cik=cik,
                accession_number=acc_dashed,
                company_name=company_name,
                form_type=form,
                filing_date=dates[i] if i < len(dates) else "",
                primary_document=primary_doc,
                document_url=doc_url,
                description=descriptions[i] if i < len(descriptions) else "",
                period_of_report=periods[i] if i < len(periods) else "",
            ))

        # Sort newest first
        results.sort(key=lambda f: f.filing_date, reverse=True)
        return results

    def _get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        stream: bool = False,
    ) -> requests.Response:
        """
        HTTP GET with EDGAR rate limiting and exponential backoff.
        EDGAR rate limit: 10 req/sec — we stay well below at ~8 req/sec.
        """
        # Enforce minimum delay between requests
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.REQUEST_DELAY_SECONDS:
            time.sleep(self.REQUEST_DELAY_SECONDS - elapsed)

        for attempt in range(max_retries):
            try:
                # EDGAR host header differs for different endpoints
                host = "www.sec.gov" if "www.sec.gov" in url else "data.sec.gov"
                self.session.headers["Host"] = host
                resp = self.session.get(url, timeout=30, stream=stream)
                self._last_request_time = time.monotonic()

                if resp.status_code == 429:
                    wait = 2 ** attempt * 5
                    logger.warning(f"EDGAR rate limited — waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue

                return resp

            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(f"Request failed ({exc}) — retry in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"All {max_retries} attempts failed for URL: {url}")
