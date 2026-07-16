import sys
import os
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.edgar_fetcher import EdgarFetcher
from src.extraction.pdf_extractor import PdfExtractor
from src.extraction.claim_detector import ClaimDetector
from src.classification.finbert_classifier import FinBertClassifier
from src.consistency.internal_checker import InternalChecker
from src.consistency.historical_checker import HistoricalChecker
from src.consistency.sbti_checker import SbtiChecker
from src.consistency.aggregator import aggregate_results

def run_test(ticker):
    print(f"--- Starting {ticker} Pipeline Test ---")
    
    fetcher = EdgarFetcher()
    filings = fetcher.fetch_recent_filings(ticker, form_type="10-K", days_back=730)
    if not filings:
        print(f"No {ticker} filings found.")
        return
        
    filing_meta = filings[0]
    print(f"Downloading: {filing_meta.form_type} for {filing_meta.company_name} ({filing_meta.filing_date})")
    
    local_path = fetcher.download_filing(filing_meta, output_dir=Path(f"data/raw/{ticker}"))
    print(f"Downloaded to {local_path}")
    
    extractor = PdfExtractor()
    doc = extractor.extract(local_path.local_path)
    doc.company_name = filing_meta.company_name
    doc.ticker = ticker
    doc.filing_year = int(filing_meta.filing_date[:4])
    print(f"Extracted {len(doc.sections)} sections.")
    
    detector = ClaimDetector()
    claims = detector.detect_from_document(doc.sections)
    print(f"Detected {len(claims)} ESG claims.")
    
    clf = FinBertClassifier()
    claim_texts = [c.text for c in claims]
    source_sections = [c.source_section for c in claims]
    results = clf.classify_batch(claim_texts, source_sections=source_sections)
    
    l1_checker = InternalChecker()
    l2_checker = HistoricalChecker()
    l3_checker = SbtiChecker()
    
    for i, (claim, result) in enumerate(zip(claims, results)): # check ALL claims
        print(f"\n[Claim {i+1}] Section: {claim.source_section}")
        print(f"Text: {claim.text}")
        print(f"Label: {result.esg_label} | NLP Conf: {result.confidence:.2f}")
        
        l1 = l1_checker.check(claim.text, claim.start_char, doc)
        l2 = l2_checker.check(claim, doc.ticker, doc.filing_year)
        l3 = l3_checker.check(claim, doc.company_name)
        
        agg = aggregate_results(
            l1=l1, l2=l2, l3=l3,
            nlp_flag=result.consistency_flag,
            esg_label=result.esg_label,
            nlp_confidence=result.confidence,
        )
        
        print(f"L1 (Internal): {l1.status if l1 else 'None'}")
        print(f"L2 (Historical): {l2.status if l2 else 'None'}")
        print(f"L3 (SBTi): {l3.status if l3 else 'None'}")
        print(f"Agg Verdict: {agg.verdict} | Score: {agg.risk_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline for given tickers")
    parser.add_argument("tickers", nargs="+", help="List of tickers to test")
    args = parser.parse_args()
    
    for ticker in args.tickers:
        run_test(ticker.upper())
