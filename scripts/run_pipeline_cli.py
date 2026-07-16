#!/usr/bin/env python
"""
scripts/run_pipeline_cli.py
==========================
CLI wrapper to run claim detection and classification pipeline in the terminal.
Allows checking claims manually for any ticker (e.g. AAPL).
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup encodings for terminal print output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
from loguru import logger

# Import config & pipeline components
from config.settings import gemini_cfg
from src.classification.finbert_classifier import FinBertClassifier
from src.extraction.claim_detector import ClaimDetector
from src.extraction.pdf_extractor import PdfExtractor
from src.ingestion.edgar_fetcher import EdgarFetcher
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    parser = argparse.ArgumentParser(description="AuditLens CLI Pipeline runner")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--quick", action="store_true", help="Enable fast ESG keyword pre-filtering before FinBERT")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates to send to Gemini per section (default: None for unlimited)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"🚀 Initializing components for ticker: {ticker}...")

    # Load components
    clf = FinBertClassifier()
    
    api_key = gemini_cfg.api_key
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not configured in .env file.")
        sys.exit(1)

    models_to_try = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.5-flash", "gemini-flash-latest"]
    ordered_models = []
    for m in [gemini_cfg.model_name] + models_to_try:
        if m and m not in ordered_models:
            ordered_models.append(m)

    llm_instances = []
    for model in ordered_models:
        llm_instances.append(ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=8192,
        ))

    primary_llm = llm_instances[0]
    if len(llm_instances) > 1:
        llm = primary_llm.with_fallbacks(llm_instances[1:])
    else:
        llm = primary_llm

    detector = ClaimDetector(
        classifier=clf,
        llm=llm,
        confidence_threshold=0.80,
        use_keyword_filter=args.quick,
        max_candidates=args.max_candidates
    )
    fetcher = EdgarFetcher()
    extractor = PdfExtractor()

    print(f"\n🔍 Fetching latest 10-K filing for {ticker} from SEC EDGAR...")
    filings = fetcher.fetch_recent_filings(ticker, form_type="10-K", days_back=365)
    if not filings:
        print(f"❌ No 10-K filing found for ticker: {ticker}")
        sys.exit(1)

    filing = filings[0]
    print(f"✅ Found filing: {filing.company_name} | {filing.form_type} | {filing.filing_date}")

    output_dir = PROJECT_ROOT / "data" / "raw" / ticker
    print(f"📥 Downloading filing to {output_dir}...")
    downloaded = fetcher.download_filing(filing, output_dir=output_dir)
    if not downloaded.download_success:
        print(f"❌ Failed to download filing: {downloaded.error}")
        sys.exit(1)

    print(f"📄 Extracting text from filing PDF...")
    doc = extractor.extract(downloaded.local_path)
    doc.company_name = filing.company_name
    doc.filing_year = int((filing.period_of_report or filing.filing_date)[:4])

    print(f"🧠 Running hybrid ESG claim extraction (FinBERT sentence filter + Gemini analysis)...")
    claims = detector.detect_from_document(doc.sections)

    if not claims:
        print("⚠️ No ESG claims were detected in this filing.")
        sys.exit(0)

    print(f"\n🎉 Successfully extracted {len(claims)} ESG claims!")
    print("=" * 80)
    print(f"{'#':<3} | {'ESG':<3} | {'Type':<12} | {'Confidence':<10} | {'Atomic Claim'}")
    print("-" * 80)
    for i, claim in enumerate(claims):
        res = clf.classify(claim.text, source_section=claim.source_section)
        print(f"{i+1:<3} | {res.esg_label:<3} | {claim.claim_type:<12} | {res.confidence:.1%} | {claim.text}")
        print(f"    👉 Original: \"{getattr(claim, 'evidence_sentence', '') or claim.text}\"")
        print(f"    📝 Context:  \"{claim.evidence}\"")
        print("-" * 80)

if __name__ == "__main__":
    main()
