"""
src/dashboard/app.py
=====================
Main Streamlit dashboard for AuditLens ESG Audit Assistant.

Pages:
  1. 📤 Upload / Search — upload PDF or enter ticker
  2. 📋 Claims Explorer — table of all extracted claims
  3. 🔍 Claim Detail   — SHAP + audit observation for one claim
  4. 💬 RAG Chatbot    — ask questions about ESG standards
  5. 📥 Export         — download structured audit report as PDF

Run locally:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

# ── Page config — MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="AuditLens — ESG Audit Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-username/esg-audit-assistant",
        "Report a bug": "https://github.com/your-username/esg-audit-assistant/issues",
        "About": "AuditLens v1.0 — AI-powered ESG audit assistant",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f172a; color: #e2e8f0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Verdict badges */
    .badge-consistent     { background:#166534; color:#bbf7d0; padding:2px 10px; border-radius:999px; font-size:0.8em; font-weight:600; }
    .badge-partial        { background:#92400e; color:#fde68a; padding:2px 10px; border-radius:999px; font-size:0.8em; font-weight:600; }
    .badge-inconsistent   { background:#991b1b; color:#fecaca; padding:2px 10px; border-radius:999px; font-size:0.8em; font-weight:600; }
    .badge-highrisk       { background:#4c1d95; color:#ddd6fe; padding:2px 10px; border-radius:999px; font-size:0.8em; font-weight:600; }

    /* Section headers */
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stMarkdown p { color: #cbd5e1; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton button:hover { opacity: 0.85; }

    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; }

    /* Expander */
    .streamlit-expanderHeader { background: #1e293b; border-radius: 8px; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def init_session():
    defaults = {
        "claims": [],
        "extracted_doc": None,
        "ticker": "",
        "company_name": "",
        "filing_year": "",
        "rag_chain": None,
        "vector_store": None,
        "classifier": None,
        "explainer": None,
        "selected_claim_idx": 0,
        "processing_complete": False,
        "audit_observations": {},   # claim_idx → AuditObservation
        "aggregate_results": {},    # claim_idx → AggregateResult
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading FinBERT classifier...")
def load_classifier():
    from src.classification.finbert_classifier import FinBertClassifier
    return FinBertClassifier()

@st.cache_resource(show_spinner="Loading embedding model...")
def load_vector_store():
    from src.rag.vector_store import VectorStore
    from config.settings import app_cfg
    index_path = app_cfg.faiss_index_path
    if index_path.exists() and (index_path / "faiss.index").exists():
        return VectorStore.load(index_path)
    else:
        st.warning(
            "⚠️ FAISS index not found. "
            "Run `python scripts/build_index.py` to build the knowledge base first. "
            "Standard compliance checks (L3) will be unavailable.",
            icon="⚠️"
        )
        return None

@st.cache_resource(show_spinner="Connecting to Gemini API...")
def load_rag_chain(_vector_store):
    if _vector_store is None:
        return None
    try:
        from src.rag.rag_chain import RagChain
        from config.settings import gemini_cfg
        return RagChain.build(_vector_store, api_key=gemini_cfg.api_key)
    except Exception as exc:
        st.error(f"Failed to connect to Gemini API: {exc}")
        return None


# ── Helper Functions ─────────────────────────────────────────────────────────

def _process_upload(uploaded_file, company_name, year, ticker, clf, vs, rc):
    """Save uploaded PDF to temp, extract, classify."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    with st.spinner("Extracting text and tables from PDF..."):
        from src.extraction.pdf_extractor import PdfExtractor
        extractor = PdfExtractor()
        doc = extractor.extract(tmp_path)
        if company_name:
            doc.company_name = company_name
        if year:
            doc.filing_year = year
        if ticker:
            doc.ticker = ticker.strip().upper()
        st.session_state.extracted_doc = doc

    _run_pipeline(doc, clf, vs, rc)
    tmp_path.unlink(missing_ok=True)


def _fetch_and_process(ticker, form_type, clf, vs, rc):
    """Fetch latest filing from EDGAR and run full pipeline."""
    from src.ingestion.edgar_fetcher import EdgarFetcher
    from src.extraction.pdf_extractor import PdfExtractor

    with st.spinner(f"Fetching {form_type} filing for {ticker} from EDGAR..."):
        fetcher = EdgarFetcher()
        filings = fetcher.fetch_recent_filings(ticker, form_type=form_type, days_back=365)

        if not filings:
            st.error(f"No {form_type} filing found for ticker '{ticker}'. Try a different ticker.")
            return

        filing = filings[0]
        st.info(f"Found filing: **{filing.company_name}** | {filing.form_type} | {filing.filing_date}")

        output_dir = Path("data/raw") / ticker
        downloaded = fetcher.download_filing(filing, output_dir)

        if not downloaded.download_success:
            st.error(f"Download failed: {downloaded.error}")
            return

        extractor = PdfExtractor()
        doc = extractor.extract(downloaded.local_path)

        doc.company_name = filing.company_name
        doc.filing_year = int((filing.period_of_report or filing.filing_date)[:4])
        doc.ticker = ticker
        st.session_state.extracted_doc = doc
        st.session_state.ticker = ticker

    _run_pipeline(doc, clf, vs, rc)


def _save_claims_to_file(processed_claims, company_name, filing_year):
    import csv
    from datetime import datetime
    import re
    
    output_dir = Path("data/classified_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comp = company_name or "Unknown_Company"
    year = filing_year or "0000"
    clean_company = re.sub(r"[^a-zA-Z0-9_\-]", "_", comp.strip())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_company}_{year}_{timestamp}.csv"
    filepath = output_dir / filename
    
    try:
        with open(filepath, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Claim ID", "Claim Text", "Source Section", "ESG Label", 
                "L1 Check Status", "L2 Check Status", "L3 Check Status", 
                "Audit Risk Score", "Verdict"
            ])
            for i, item in enumerate(processed_claims):
                c = item["claim"]
                r = item["classification"]
                ag = item["agg_result"]
                l1 = item.get("l1_result")
                l2 = item.get("l2_result")
                l3 = item.get("l3_result")
                
                writer.writerow([
                    i + 1,
                    c.text,
                    c.source_section,
                    r.esg_label,
                    l1.status if l1 else "—",
                    l2.status if l2 else "—",
                    l3.status if l3 else "—",
                    f"{ag.risk_score:.3f}",
                    ag.verdict
                ])
        logger.info(f"Saved run results to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save claims to CSV: {e}")


def _trim_sections(sections: dict) -> dict:
    """
    Trim the 'general' section to at most 60,000 chars to prevent the full
    10-K body (financial + boilerplate) from being sent through FinBERT.
    ESG-specific sections (environmental/social/governance) are kept as-is.
    """
    MAX_GENERAL_CHARS = 60_000
    trimmed = {}
    for name, text in sections.items():
        if name == "general" and len(text) > MAX_GENERAL_CHARS:
            trimmed[name] = text[:MAX_GENERAL_CHARS]
        else:
            trimmed[name] = text
    return trimmed


def _run_pipeline(doc, clf, vs, rc):
    """Run claim detection + classification + consistency checks on extracted doc."""
    from src.extraction.claim_detector import ClaimDetector

    progress = st.progress(0)

    with st.spinner("Detecting ESG claims (FinBERT + Gemini)..."):
        if rc is None or rc.llm is None:
            st.error("❌ Gemini LLM is not configured (check your GEMINI_API_KEY in .env). Claim detection cannot proceed.")
            st.stop()

        # Key performance settings:
        # - use_keyword_filter=True  → only ESG-keyword sentences go to FinBERT (10x speedup)
        # - max_candidates=150       → cap Gemini batches even if many candidates pass filter
        detector = ClaimDetector(
            classifier=clf,
            llm=rc.llm,
            use_keyword_filter=True,
            max_candidates=150,
        )

        # Trim general section so we don't run FinBERT over the full financial 10-K body
        sections = _trim_sections(doc.sections)
        claims = detector.detect_from_document(sections)
        if not claims:
            st.warning("No ESG claims detected. The document may not contain verifiable sustainability statements.")
            return

    # FinBERT classification is already done inside ClaimDetector.detect_from_document.
    # Each claim already has confidence + esg_label from FinBERT.
    # We reuse those results directly instead of re-running FinBERT a second time.
    from src.classification.finbert_classifier import ClassificationResult
    results = [
        ClassificationResult(
            claim_text=c.text,
            esg_label=c.esg_label,
            risk_score=0.35 if c.esg_label == "E" else (0.30 if c.esg_label == "S" else (0.55 if c.esg_label == "G" else 0.70)),
            consistency_flag="LIKELY_CONSISTENT" if c.confidence >= 0.85 else ("NEEDS_REVIEW" if c.confidence >= 0.65 else "HIGH_RISK"),
            confidence=c.confidence,
        )
        for c in claims
    ]

    from src.consistency.verifier import VerificationPipeline
    pipeline = VerificationPipeline(vector_store=vs)

    processed_claims = []
    for i, (claim, result) in enumerate(zip(claims, results)):
        progress.progress((i + 1) / len(claims), text=f"Verifying claim {i+1}/{len(claims)}...")

        # Run verification pipeline (L1 Internal, L2 Historical, L3 Standards Compliance)
        res = pipeline.verify_claim(claim, doc)

        processed_claims.append({
            "claim": claim,
            "classification": result,
            "l1_result": res["l1"],
            "l2_result": res["l2"],
            "l3_result": res["l3"],
            "agg_result": res["agg_result"],
            "shap_result": None,
            "audit_observation": None,
        })

    progress.empty()

    # Automatically save classified claims to a CSV file in data/classified_runs
    _save_claims_to_file(processed_claims, doc.company_name, doc.filing_year)

    st.session_state.claims = processed_claims
    st.session_state.company_name = doc.company_name
    st.session_state.filing_year = doc.filing_year
    st.session_state.processing_complete = True
    st.success(f"✅ {len(processed_claims)} claims classified")
    st.rerun()


def _show_summary_metrics():
    """Show top-line metrics after processing."""
    claims_data = st.session_state.claims
    total      = len(claims_data)
    high_risk  = sum(1 for c in claims_data if c["agg_result"].verdict == "HIGH_RISK")
    inconsist  = sum(1 for c in claims_data if c["agg_result"].verdict == "INCONSISTENT")
    partial    = sum(1 for c in claims_data if c["agg_result"].verdict == "PARTIALLY_CONSISTENT")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims",        total)
    col2.metric("Partially Consistent ⚠️", partial)
    col3.metric("Inconsistent ❌",      inconsist)
    col4.metric("High Risk 🚨",         high_risk)


def _generate_pdf_report(company: str, year: str) -> bytes:
    """Generate a professional PDF audit report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    import io
    from datetime import date

    W, H = A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=20*mm, rightMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    story = []

    NAVY   = HexColor("#1e3a8a")
    SLATE  = HexColor("#334155")
    LIGHT  = HexColor("#f8fafc")
    ALT    = HexColor("#e2e8f0")
    GREEN  = HexColor("#166534")
    AMBER  = HexColor("#92400e")
    RED    = HexColor("#991b1b")
    PURPLE = HexColor("#4c1d95")

    VERDICT_BG = {
        "CONSISTENT":           GREEN,
        "PARTIALLY_CONSISTENT": AMBER,
        "INCONSISTENT":         RED,
        "HIGH_RISK":            PURPLE,
    }

    # ── Styles ──────────────────────────────────────────────────────────────
    h1 = ParagraphStyle("H1", fontSize=22, textColor=white,
                        fontName="Helvetica-Bold", spaceAfter=4)
    h2 = ParagraphStyle("H2", fontSize=14, textColor=NAVY,
                        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6,
                        borderPad=4)
    body = ParagraphStyle("Body", fontSize=9, textColor=HexColor("#1e293b"),
                          leading=14, spaceAfter=4)
    caption = ParagraphStyle("Cap", fontSize=8, textColor=HexColor("#64748b"),
                             leading=11)
    claim_style = ParagraphStyle("Claim", fontSize=9, textColor=HexColor("#1e293b"),
                                 leading=14, leftIndent=8, spaceAfter=2)

    # ── Cover banner ────────────────────────────────────────────────────────
    banner_data = [[
        Paragraph("🔍  AuditLens — ESG Audit Report", h1),
    ]]
    banner = Table(banner_data, colWidths=[W - 40*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("PADDING",    (0,0), (-1,-1), 14),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6))

    # Sub-header line
    story.append(Paragraph(
        f"<b>{company}</b> &nbsp;·&nbsp; Filing Year: <b>{year}</b> &nbsp;·&nbsp; "
        f"Generated: <b>{date.today().strftime('%d %b %Y')}</b>",
        ParagraphStyle("sub", fontSize=10, textColor=SLATE, spaceAfter=4),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=ALT, spaceAfter=10))

    # ── Executive summary table ──────────────────────────────────────────────
    claims = st.session_state.claims
    verdicts = [c["agg_result"].verdict for c in claims]

    esg_counts = {"E": 0, "S": 0, "G": 0, "MIXED": 0}
    for c in claims:
        lbl = c["classification"].esg_label
        esg_counts[lbl] = esg_counts.get(lbl, 0) + 1

    flagged = [c for c in claims if c["agg_result"].verdict in ("INCONSISTENT", "HIGH_RISK")]
    avg_audit_risk = (
        sum(c["agg_result"].risk_score for c in claims) / len(claims)
        if claims else 0.0
    )

    story.append(Paragraph("Executive Summary", h2))

    summary_data = [
        ["Metric", "Value"],
        ["Total ESG Claims Analysed",    str(len(claims))],
        ["Claims Flagged for Review",    str(len(flagged))],
        ["Average Audit Risk Score",     f"{avg_audit_risk:.2f} / 1.00"],
        ["✅  Consistent",               str(verdicts.count("CONSISTENT"))],
        ["⚠️  Partially Consistent",     str(verdicts.count("PARTIALLY_CONSISTENT"))],
        ["❌  Inconsistent",             str(verdicts.count("INCONSISTENT"))],
        ["🚨  High Risk",                str(verdicts.count("HIGH_RISK"))],
        ["Environmental (E) claims",     str(esg_counts["E"])],
        ["Social (S) claims",            str(esg_counts["S"])],
        ["Governance (G) claims",        str(esg_counts["G"])],
        ["Mixed claims",                 str(esg_counts["MIXED"])],
    ]

    tbl_styles = [
        ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, ALT]),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
        ("PADDING",     (0, 0), (-1, -1), 7),
        ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
    ]
    t = Table(summary_data, colWidths=[120*mm, 50*mm])
    t.setStyle(TableStyle(tbl_styles))
    story.append(t)
    story.append(Spacer(1, 16))

    # ── Flagged claims section ───────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=ALT, spaceAfter=8))
    story.append(Paragraph("Flagged Claims Requiring Auditor Review", h2))

    if not flagged:
        story.append(Paragraph("No claims were flagged as Inconsistent or High Risk.", body))
    else:
        for i, item in enumerate(flagged):
            agg = item["agg_result"]
            clf = item["classification"]
            obs = item.get("audit_observation")
            verdict_bg = VERDICT_BG.get(agg.verdict, SLATE)

            # Claim header row
            claim_num = claims.index(item) + 1
            header_data = [[
                Paragraph(
                    f"<b>Claim #{claim_num}</b> &nbsp; "
                    f"ESG: {clf.esg_label} &nbsp;·&nbsp; "
                    f"Section: {item['claim'].source_section.title()} &nbsp;·&nbsp; "
                    f"Audit Risk: <b>{agg.risk_score:.2f}</b> &nbsp;·&nbsp; "
                    f"Verdict: <b>{agg.verdict.replace('_', ' ')}</b>",
                    ParagraphStyle("ch", fontSize=9, textColor=white, fontName="Helvetica-Bold"),
                ),
            ]]
            ch = Table(header_data, colWidths=[W - 40*mm])
            ch.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), verdict_bg),
                ("PADDING",    (0,0), (-1,-1), 8),
            ]))

            # Claim text
            claim_text = Paragraph(item["claim"].text, claim_style)

            # Summary note
            summary_note = Paragraph(
                f"<i>Consistency summary:</i> {agg.summary}", caption
            )

            # Audit observation (if generated)
            obs_para = None
            if obs:
                obs_para = Paragraph(
                    f"<b>Audit Observation:</b> {obs.structured_note[:500]}", body
                )
                if obs.standards_cited:
                    obs_para = Paragraph(
                        f"<b>Audit Observation:</b> {obs.structured_note[:400]}<br/>"
                        f"<i>Standards: {', '.join(obs.standards_cited)}</i>", body
                    )

            block = [ch, Spacer(1, 4), claim_text, summary_note]
            if obs_para:
                block.append(obs_para)
            block.append(Spacer(1, 10))

            story.append(KeepTogether(block))

    # ── Footer note ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ALT))
    story.append(Paragraph(
        "This report was generated by AuditLens v1.0. Audit Risk scores are computed "
        "by a multi-level consistency pipeline (L1 internal · L2 historical · L3 GRI/TCFD/ISSB). "
        "All flagged claims should be reviewed by a qualified auditor before conclusions are drawn.",
        ParagraphStyle("footer", fontSize=7, textColor=HexColor("#94a3b8"), leading=10, spaceBefore=6),
    ))

    doc.build(story)
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 AuditLens")
    st.markdown("*AI-powered ESG audit assistant*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📤 Upload / Search", "📋 Claims Explorer",
         "🔍 Claim Detail", "💬 Standards Chatbot", "📥 Export Report"],
        label_visibility="collapsed",
    )

    st.divider()

    # System status
    st.markdown("**System Status**")
    clf = load_classifier()
    vs  = load_vector_store()
    rc  = load_rag_chain(vs)

    st.markdown(f"{'🟢' if clf else '🔴'} FinBERT: {'Loaded' if clf else 'Not loaded'}")
    st.markdown(f"{'🟢' if vs  else '🟡'} FAISS Index: {'Ready' if vs else 'Not built'}")
    st.markdown(f"{'🟢' if rc  else '🟡'} Gemini RAG: {'Ready' if rc else 'No API key'}")

    st.divider()
    st.caption("v1.0 | AuditLens ESG Audit Assistant")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Upload / Search
# ═════════════════════════════════════════════════════════════════════════════

if page == "📤 Upload / Search":
    st.title("📤 ESG Filing Analysis")
    st.markdown("Upload an ESG PDF filing or enter a company ticker to fetch from EDGAR.")

    tab_upload, tab_ticker = st.tabs(["📎 Upload PDF", "🏢 Search by Ticker"])

    # ── Upload PDF tab
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload ESG Report PDF",
            type=["pdf"],
            help="Annual reports, sustainability reports, 10-K filings",
        )
        company_override = st.text_input("Company Name (optional)", placeholder="e.g. Apple Inc.")
        year_override    = st.text_input("Filing Year (optional)",   placeholder="e.g. 2023")
        ticker_override  = st.text_input("Ticker Symbol (optional)", placeholder="e.g. AAPL")

        if uploaded and st.button("🚀 Analyse Filing", key="analyse_upload"):
            _process_upload(uploaded, company_override, year_override, ticker_override, clf, vs, rc)

    # ── Ticker search tab
    with tab_ticker:
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input(
                "Stock Ticker", placeholder="e.g. AAPL, MSFT, TSLA",
                key="ticker_input",
            ).upper()
        with col2:
            form_type = st.selectbox("Form Type", ["10-K", "20-F", "DEF 14A"], key="form_sel")

        if ticker and st.button("🔎 Fetch from EDGAR", key="fetch_edgar"):
            _fetch_and_process(ticker, form_type, clf, vs, rc)

    # ── Processing results preview
    if st.session_state.processing_complete and st.session_state.claims:
        st.divider()
        st.success(f"✅ Analysis complete — {len(st.session_state.claims)} claims detected")
        _show_summary_metrics()
# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Claims Explorer
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📋 Claims Explorer":
    st.title("📋 Claims Explorer")

    if not st.session_state.claims:
        st.info("👈 Go to **Upload / Search** to analyse a filing first.")
        st.stop()

    company = st.session_state.company_name or "Unknown Company"
    year    = st.session_state.filing_year or "—"
    st.markdown(f"**{company}** | Filing year: **{year}** | {len(st.session_state.claims)} claims")

    # Build display DataFrame
    rows = []
    for i, item in enumerate(st.session_state.claims):
        c  = item["claim"]
        r  = item["classification"]
        ag = item["agg_result"]
        l1 = item.get("l1_result")
        l2 = item.get("l2_result")
        l3 = item.get("l3_result")
        
        rows.append({
            "#":           i + 1,
            "Claim":       c.text,
            "Section":     c.source_section.title(),
            "ESG":         r.esg_label,
            "L1 Check":    l1.status if l1 else "—",
            "L2 Check":    l2.status if l2 else "—",
            "L3 Check":    l3.status if l3 else "—",
            "Audit Risk":  round(ag.risk_score, 2),
            "Verdict":     ag.verdict,
        })

    df = pd.DataFrame(rows)

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        section_filter = st.multiselect(
            "Section", options=df["Section"].unique().tolist(), default=[],
        )
    with col2:
        esg_filter = st.multiselect(
            "ESG Label", options=["E", "S", "G", "MIXED"], default=[],
        )
    with col3:
        verdict_filter = st.multiselect(
            "Verdict", options=df["Verdict"].unique().tolist(), default=[],
        )

    filtered_df = df.copy()
    if section_filter:
        filtered_df = filtered_df[filtered_df["Section"].isin(section_filter)]
    if esg_filter:
        filtered_df = filtered_df[filtered_df["ESG"].isin(esg_filter)]
    if verdict_filter:
        filtered_df = filtered_df[filtered_df["Verdict"].isin(verdict_filter)]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Audit Risk": st.column_config.ProgressColumn(
                "Audit Risk", help="Weighted audit risk from multi-level consistency checks",
                min_value=0, max_value=1, format="%.2f"
            ),
        },
    )

    # Quick navigation to claim detail
    st.divider()
    selected = st.number_input(
        "Go to Claim # for detailed analysis",
        min_value=1, max_value=len(st.session_state.claims), value=1, step=1,
    )
    if st.button("🔍 Analyse this claim"):
        st.session_state.selected_claim_idx = int(selected) - 1
        st.rerun()





# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Claim Detail
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Claim Detail":
    from src.dashboard.components.claim_card import render_claim_card
    from src.dashboard.components.shap_chart import render_shap_chart

    st.title("🔍 Claim Detail")

    if not st.session_state.claims:
        st.info("👈 Go to **Upload / Search** to analyse a filing first.")
        st.stop()

    idx = st.session_state.selected_claim_idx
    idx = st.number_input(
        f"Claim # (1–{len(st.session_state.claims)})",
        min_value=1, max_value=len(st.session_state.claims),
        value=idx + 1, step=1,
    ) - 1
    st.session_state.selected_claim_idx = idx

    item = st.session_state.claims[idx]

    # ── Claim card
    render_claim_card(item, idx)
    st.divider()

    # ── SHAP explanation
    col_shap, col_obs = st.columns([1, 1])

    with col_shap:
        st.markdown("### 🧠 SHAP Explanation")
        shap_result = item.get("shap_result")

        if shap_result is None:
            # Compute on demand
            if st.button("Generate SHAP Explanation (slow on CPU ~5s)"):
                from src.classification.shap_explainer import ShapExplainer
                with st.spinner("Computing SHAP values..."):
                    explainer = ShapExplainer(classifier=load_classifier())
                    shap_result = explainer.explain(
                        item["claim"].text,
                        prediction_label=item["classification"].esg_label,
                    )
                    st.session_state.claims[idx]["shap_result"] = shap_result
                    st.rerun()
        else:
            render_shap_chart(shap_result)

    with col_obs:
        st.markdown("### 📝 Audit Observation")
        obs = item.get("audit_observation")

        if obs is None:
            auditor_q = st.text_input(
                "Optional auditor question",
                placeholder="e.g. What GRI disclosure is required for this claim?",
                key=f"q_{idx}",
            )
            if st.button("Generate Audit Observation (Gemini)", key=f"gen_{idx}"):
                if rc is None:
                    st.error("RAG chain not available — check Gemini API key in .env")
                else:
                    shap_narrative = (
                        shap_result.narrative if shap_result else "No SHAP explanation available."
                    )
                    with st.spinner("Generating audit observation with Gemini..."):
                        obs = rc.generate_audit_observation(
                            claim=item["claim"].text,
                            shap_narrative=shap_narrative,
                            question=auditor_q or None,
                        )
                        st.session_state.claims[idx]["audit_observation"] = obs
                        st.rerun()
        else:
            if obs.low_confidence_flag:
                st.warning("⚠️ Low retrieval confidence — standard match may be imprecise.")
            st.markdown(obs.structured_note)

            if obs.standards_cited:
                st.markdown(f"**Standards cited:** {', '.join(obs.standards_cited)}")

            # ── Consistency check details
            st.divider()
            st.markdown("### ✅ Consistency Check Levels")
            agg = item["agg_result"]
            for lvl, result in agg.level_results.items():
                status_icon = {"SUPPORTED": "✅", "PARTIALLY_SUPPORTED": "⚠️", "UNSUPPORTED": "❌", "SKIPPED": "⏭️"}.get(result.status, "❓")
                with st.expander(f"L{lvl} {status_icon} {result.status}"):
                    st.markdown(result.note)
                    if result.evidence:
                        st.code(result.evidence[:400], language=None)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RAG Chatbot
# ═════════════════════════════════════════════════════════════════════════════

elif page == "💬 Standards Chatbot":
    st.title("💬 ESG Standards Chatbot")
    st.markdown("Ask questions about GRI, TCFD, SASB, ISSB, and CSRD requirements.")

    if rc is None:
        st.error("RAG chain not available. Check your GEMINI_API_KEY and FAISS index.")
        st.stop()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New message
    if prompt := st.chat_input("Ask about ESG standards..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching standards knowledge base..."):
                answer = rc.answer_question(prompt)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Export Report
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📥 Export Report":
    st.title("📥 Export Audit Report")

    if not st.session_state.claims:
        st.info("👈 Go to **Upload / Search** to analyse a filing first.")
        st.stop()

    company = st.session_state.company_name or "Unknown Company"
    year    = st.session_state.filing_year or "N/A"
    flagged = [c for c in st.session_state.claims if c["agg_result"].verdict in ("INCONSISTENT", "HIGH_RISK")]

    st.markdown(f"**{company} — {year} ESG Filing Audit**")
    st.metric("Total Claims Analysed", len(st.session_state.claims))
    st.metric("Flagged for Review", len(flagged))

    if st.button("📄 Generate PDF Report"):
        pdf_bytes = _generate_pdf_report(company, year)
        st.download_button(
            "⬇️ Download Audit Report PDF",
            data=pdf_bytes,
            file_name=f"AuditLens_{company.replace(' ', '_')}_{year}.pdf",
            mime="application/pdf",
        )

    # JSON export
    if st.button("📊 Export Claims JSON"):
        export_data = [
            {
                "claim": c["claim"].text,
                "section": c["claim"].source_section,
                "esg_label": c["classification"].esg_label,
                "nlp_confidence": c["classification"].confidence,
                "audit_risk_score": c["agg_result"].risk_score,
                "verdict": c["agg_result"].verdict,
                "audit_note": c["audit_observation"].structured_note if c["audit_observation"] else "",
            }
            for c in st.session_state.claims
        ]
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"AuditLens_{company}_{year}_claims.json",
            mime="application/json",
        )
# (Drift Monitor page removed — requires active MLOps pipeline)
