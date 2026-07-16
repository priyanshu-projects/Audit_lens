"""
tests/conftest.py
==================
Shared pytest fixtures for AuditLens test suite.
"""

from __future__ import annotations

from pathlib import Path
import pytest


# ── Sample ESG claim texts ───────────────────────────────────────────────────

SAMPLE_CLAIMS = [
    "We reduced our Scope 1 GHG emissions by 40% compared to our 2019 baseline.",
    "Our renewable energy target is 100% by 2030.",
    "Employee turnover rate was 8.2%, down from 11.5% in the prior year.",
    "The Board conducted an annual review of climate risk management.",
    "We committed to Science Based Targets aligned with a 1.5°C pathway.",
    "Water consumption intensity reduced by 22% per unit of production.",
    "Zero fatal accidents were recorded across all operations in 2023.",
    "We achieved LEED Gold certification for our headquarters building.",
]

NON_ESG_SENTENCES = [
    "The company was founded in 1993.",
    "Please see our financial statements for detailed information.",
    "This report has been prepared in accordance with GAAP.",
]


@pytest.fixture
def sample_claim_texts() -> list[str]:
    return SAMPLE_CLAIMS


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal PDF-like text file for testing (not a real PDF)."""
    pdf_file = tmp_path / "test_filing.txt"
    pdf_file.write_text(
        "Annual ESG Report 2023\n\n"
        + "\n\n".join(SAMPLE_CLAIMS)
        + "\n\nScope 1 emissions data:\n"
        + "Year | Emissions (tCO2e)\n2019 | 850,000\n2023 | 510,000"
    )
    return pdf_file


@pytest.fixture
def mock_extracted_doc(tmp_path: Path):
    """Create a mock ExtractedDocument for testing."""
    from src.extraction.pdf_extractor import ExtractedDocument
    raw_text = "\n\n".join(SAMPLE_CLAIMS)
    raw_text += "\n\n2019 | 850,000\n2023 | 510,000\n40% reduction verified."
    return ExtractedDocument(
        source_path=tmp_path / "test.pdf",
        company_name="Test Corp",
        filing_year="2023",
        raw_text=raw_text,
        pages=[raw_text],
        sections={
            "environmental": SAMPLE_CLAIMS[0] + "\n" + SAMPLE_CLAIMS[4] + "\n" + SAMPLE_CLAIMS[5],
            "social": SAMPLE_CLAIMS[2] + "\n" + SAMPLE_CLAIMS[6],
            "governance": SAMPLE_CLAIMS[3],
        },
    )


@pytest.fixture
def mock_claims(mock_extracted_doc):
    """Detect claims from the mock document."""
    from src.extraction.claim_detector import ClaimDetector
    detector = ClaimDetector()
    return detector.detect_from_document(mock_extracted_doc.sections)


@pytest.fixture
def mock_vector_store(tmp_path: Path):
    """Create a minimal VectorStore with test documents."""
    from src.rag.vector_store import VectorStore
    test_chunks = [
        "GRI 305-1 requires companies to disclose Scope 1 GHG emissions in metric tonnes CO2 equivalent.",
        "TCFD recommends disclosure of climate-related risks across short, medium, and long time horizons.",
        "GRI 401-1 requires reporting of new employee hires and employee turnover by age group and gender.",
        "ISSB IFRS S2 requires disclosure of Scope 1, 2, and 3 greenhouse gas emissions.",
        "GRI 302-1 requires disclosure of energy consumption from renewable and non-renewable sources.",
    ]
    metadatas = [
        {"source_doc": "GRI_305.pdf", "standard": "GRI", "section": "305-1", "page_number": 1},
        {"source_doc": "TCFD.pdf", "standard": "TCFD", "section": "Risk Disclosure", "page_number": 5},
        {"source_doc": "GRI_401.pdf", "standard": "GRI", "section": "401-1", "page_number": 2},
        {"source_doc": "IFRS_S2.pdf", "standard": "ISSB", "section": "GHG Disclosure", "page_number": 8},
        {"source_doc": "GRI_302.pdf", "standard": "GRI", "section": "302-1", "page_number": 3},
    ]
    store = VectorStore()
    store.build_from_texts(test_chunks, metadatas)
    return store
