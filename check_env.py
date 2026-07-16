"""Quick environment + module check for AuditLens."""
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

failures = []

def chk(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        failures.append(name)
        print(f"  FAIL {name}: {e}")

# Core
chk("spacy",          lambda: __import__("spacy"))
chk("spacy model",    lambda: __import__("spacy").load("en_core_web_sm"))
chk("plotly",         lambda: __import__("plotly"))
chk("langchain",      lambda: __import__("langchain"))
chk("langchain_google_genai", lambda: __import__("langchain_google_genai"))
chk("pdfplumber",     lambda: __import__("pdfplumber"))
chk("loguru",         lambda: __import__("loguru"))
chk("reportlab",      lambda: __import__("reportlab"))
chk("faiss",          lambda: __import__("faiss"))
chk("sentence_transformers", lambda: __import__("sentence_transformers"))
chk("transformers",   lambda: __import__("transformers"))
chk("torch",          lambda: __import__("torch"))
chk("shap",           lambda: __import__("shap"))
chk("pandas",         lambda: __import__("pandas"))
chk("streamlit",      lambda: __import__("streamlit"))
chk("lxml",           lambda: __import__("lxml"))
chk("beautifulsoup4", lambda: __import__("bs4"))
chk("rank_bm25",      lambda: __import__("rank_bm25"))
chk("pydantic_settings", lambda: __import__("pydantic_settings"))

print("\n--- Project modules ---")
chk("PdfExtractor",        lambda: __import__("src.extraction.pdf_extractor", fromlist=["PdfExtractor"]))
chk("ClaimDetector",       lambda: __import__("src.extraction.claim_detector", fromlist=["ClaimDetector"]))
chk("EdgarFetcher",        lambda: __import__("src.ingestion.edgar_fetcher", fromlist=["EdgarFetcher"]))
chk("FinBertClassifier",   lambda: __import__("src.classification.finbert_classifier", fromlist=["FinBertClassifier"]))
chk("InternalChecker",     lambda: __import__("src.consistency.internal_checker", fromlist=["InternalChecker"]))
chk("HistoricalChecker",   lambda: __import__("src.consistency.historical_checker", fromlist=["HistoricalChecker"]))
chk("SbtiChecker",         lambda: __import__("src.consistency.sbti_checker", fromlist=["SbtiChecker"]))
chk("aggregate_results",   lambda: __import__("src.consistency.aggregator", fromlist=["aggregate_results"]))
chk("VectorStore",         lambda: __import__("src.rag.vector_store", fromlist=["VectorStore"]))
chk("RagChain",            lambda: __import__("src.rag.rag_chain", fromlist=["RagChain"]))
chk("ShapExplainer",       lambda: __import__("src.classification.shap_explainer", fromlist=["ShapExplainer"]))

print(f"\n{'='*50}")
if failures:
    print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✅")
