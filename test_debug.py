import sys
from pathlib import Path
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import gemini_cfg
from src.classification.finbert_classifier import FinBertClassifier
from src.extraction.claim_detector import ClaimDetector
from src.extraction.pdf_extractor import PdfExtractor
from src.ingestion.edgar_fetcher import EdgarFetcher
from langchain_google_genai import ChatGoogleGenerativeAI

logger.info("Initializing components...")
api_key = gemini_cfg.api_key
print(f"API Key present: {bool(api_key)}")

clf = FinBertClassifier()
llm = ChatGoogleGenerativeAI(
    model=gemini_cfg.model_name,
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=8192
)

detector = ClaimDetector(classifier=clf, llm=llm, confidence_threshold=0.8, use_keyword_filter=True, max_candidates=150)
fetcher = EdgarFetcher()
extractor = PdfExtractor()

ticker = "TSLA"
print(f"\n--- Fetching {ticker} ---")
filings = fetcher.fetch_recent_filings(ticker, form_type="10-K", days_back=365)
if not filings:
    print("No filing found!")
    sys.exit(1)

filing = filings[0]
print(f"Filing found: {filing.company_name} | {filing.filing_date}")

output_dir = PROJECT_ROOT / "data" / "raw" / ticker
downloaded = fetcher.download_filing(filing, output_dir=output_dir)
print(f"Downloaded: {downloaded.local_path} (exists={downloaded.local_path.exists()})")

doc = extractor.extract(downloaded.local_path)
print(f"Extracted pages={doc.page_count}, raw_text_len={len(doc.raw_text):,}, sections={list(doc.sections.keys())}")

for sec_name, sec_text in doc.sections.items():
    print(f"Section '{sec_name}': {len(sec_text):,} chars")

print("\n--- Running ClaimDetector ---")
claims = detector.detect_from_document(doc.sections)
print(f"\nResult: Extracted {len(claims)} claims!")
for i, c in enumerate(claims[:5]):
    print(f"{i+1}. [{c.esg_label}] {c.text}")
