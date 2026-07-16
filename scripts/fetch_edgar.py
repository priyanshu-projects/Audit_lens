"""
scripts/fetch_edgar.py
========================
Daily ingestion script — fetches new ESG filings from EDGAR.
Called by GCP Cloud Scheduler and DVC pipeline.

Usage:
    python scripts/fetch_edgar.py --tickers config/tickers.txt --days-back 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion.edgar_fetcher import EdgarFetcher
from config.settings import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Fetch new EDGAR ESG filings")
    parser.add_argument("--tickers", type=Path, default=Path("config/tickers.txt"))
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--form-type", default="10-K")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.read_text().splitlines() if t.strip()]
    logger.info(f"Fetching {args.form_type} filings for {len(tickers)} tickers (last {args.days_back} days)")

    fetcher = EdgarFetcher()
    output_dir = DATA_DIR / "raw"

    total_downloaded = 0
    for ticker in tickers:
        filings = fetcher.fetch_recent_filings(ticker, form_type=args.form_type, days_back=args.days_back)
        for filing in filings:
            result = fetcher.download_filing(filing, output_dir=output_dir / ticker)
            if result.download_success:
                total_downloaded += 1
                logger.info(f"  ✅ {ticker} — {filing.filing_date} — {result.local_path.name}")
            else:
                logger.warning(f"  ❌ {ticker} — {filing.filing_date} — {result.error}")

    logger.success(f"Ingestion complete: {total_downloaded} filings downloaded")


if __name__ == "__main__":
    main()
