"""
src/extraction/pdf_extractor.py
================================
Extracts text and tables from ESG PDF filings using pdfplumber.

Usage:
    extractor = PdfExtractor()
    doc = extractor.extract(Path("data/raw/aapl_10k.pdf"))
    print(doc.raw_text[:500])
    print(doc.sections["environmental"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from loguru import logger


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TableData:
    """A table extracted from a PDF page."""
    page_number: int
    headers: list[str]
    rows: list[list[str]]
    raw: list[list[Optional[str]]] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert table to pipe-separated text for claim cross-referencing."""
        lines = [" | ".join(str(h) for h in self.headers)]
        lines += [" | ".join(str(c) if c else "" for c in row) for row in self.rows]
        return "\n".join(lines)


@dataclass
class ExtractedDocument:
    """Full extraction result for a single PDF filing."""
    source_path: Path
    company_name: str = ""
    ticker: str = ""                             # stock ticker, e.g. "AAPL"
    filing_year: str = ""
    raw_text: str = ""                           # full text, all pages
    pages: list[str] = field(default_factory=list)   # per-page text
    tables: list[TableData] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)  # keyed by section name
    page_count: int = 0
    extraction_errors: list[str] = field(default_factory=list)


# ── ESG section detection keywords ───────────────────────────────────────────
# IMPORTANT: These must be specific enough to NOT match financial MD&A sections.
# The 10-K has both financial and ESG content — we only want ESG sections.

# Headings that indicate a page is part of a FINANCIAL section (skip for ESG)
_FINANCIAL_SECTION_BLOCKLIST = re.compile(
    r"\b(?:"
    r"management.{0,10}discussion|MD&A|results of operations|"
    r"liquidity and capital|financial condition|"
    r"net sales|gross margin|operating income|net income|revenue|"
    r"segment information|geographic information|"
    r"selected financial|consolidated statements|"
    r"balance sheet|income statement|cash flow|"
    r"risk factors|forward.looking|legal proceedings|"
    r"market for registrant|quantitative.*disclosures about market risk"
    r")\b",
    re.IGNORECASE,
)

_SECTION_PATTERNS: dict[str, list[str]] = {
    "environmental": [
        r"\benvironmental\b", r"\bclimate change\b", r"\bclimate risk\b",
        r"\bemission(?:s)?\b", r"\bcarbon\b", r"\bclean energy\b",
        r"\brenewable energy\b", r"\benergy consumption\b",
        r"\bwater\b.{0,20}\bsustainab\b", r"\bwaste\b.{0,20}\bsustainab\b",
        r"\bbiodiversity\b", r"\bgreenhous\b", r"\bscope [123]\b",
        r"\bnet.?zero\b", r"\bcarbon.?neutral\b",
    ],
    "social": [
        r"\bhuman capital\b", r"\bworkforce\b.{0,30}\bdivers\b",
        r"\bdiversity.*equity.*inclusion\b", r"\bDEI\b",
        r"\bemployee.*safety\b", r"\bworkplace.*safety\b",
        r"\bhuman rights\b", r"\blabor.*standard\b",
        r"\bsupply chain.*labor\b", r"\bcommunity.*invest\b",
        r"\bsocial impact\b", r"\bphilanthrop\b",
        r"\bdata privacy\b.{0,40}\bcommit\b",
    ],
    "governance": [
        r"\bcorporate governance\b", r"\bboard of directors\b",
        r"\baudit committee\b", r"\bexecutive compensation\b",
        r"\banti.corruption\b", r"\banti.bribery\b",
        r"\bwhistleblower\b", r"\bcode of (?:ethics|conduct)\b",
        r"\bGRI\b.{0,30}\bindex\b", r"\bsustainability.*governance\b",
        r"\btransparency.*report\b",
    ],
    "general": [],  # catch-all
}



# ── Main class ────────────────────────────────────────────────────────────────

class PdfExtractor:
    """
    Extracts structured content from ESG PDF filings.

    Responsibilities:
    - Page-by-page text extraction (pdfplumber)
    - Table detection and structured parsing
    - Heuristic section classification (E/S/G)
    - Company name and year extraction from metadata

    Note: Very large PDFs (500+ pages) are chunked — only the first
    200 pages are processed by default to keep inference time reasonable.
    """

    MAX_PAGES = 200

    def __init__(self, max_pages: int = MAX_PAGES) -> None:
        self.max_pages = max_pages

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """
        Main entry point. Extracts all content from a PDF filing.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractedDocument with raw_text, pages, tables, and sections
        """
        logger.info(f"Extracting: {pdf_path.name}")
        doc = ExtractedDocument(source_path=pdf_path)

        if not pdf_path.exists():
            doc.extraction_errors.append(f"File not found: {pdf_path}")
            logger.error(f"PDF not found: {pdf_path}")
            return doc

        is_html = pdf_path.suffix.lower() in (".htm", ".html")
        if is_html:
            try:
                from bs4 import BeautifulSoup
                import re as _re

                logger.info(f"Extracting HTML/iXBRL filing: {pdf_path.name}")
                raw_html = pdf_path.read_text(encoding="utf-8", errors="replace")
                soup = BeautifulSoup(raw_html, "lxml")

                # Remove noise: scripts, styles, XBRL hidden metadata
                for tag in soup(["script", "style", "ix:hidden",
                                  "ix:header", "link", "meta"]):
                    tag.decompose()

                # Extract all visible text
                full_text = soup.get_text(separator="\n", strip=True)

                # Collapse excessive blank lines
                full_text = _re.sub(r"\n{3,}", "\n\n", full_text)

                # Simulate pages by splitting into ~3 000-char chunks
                chunk_size = 3000
                chunks = [
                    full_text[i: i + chunk_size]
                    for i in range(0, len(full_text), chunk_size)
                ]
                chunks = chunks[: self.max_pages]

                doc.pages = chunks
                doc.page_count = len(chunks)
                doc.raw_text = full_text
                doc.sections = self._classify_sections(chunks)
                doc.filing_year = self._extract_year(full_text)

                # Extract company name: prefer explicit dei:EntityRegistrantName XBRL element,
                # then <title>, then first h1/h2 heading, then filename fallback
                entity_tag = soup.find(attrs={"name": "dei:EntityRegistrantName"})
                if not entity_tag:
                    entity_tag = soup.find("ix:nonfraction", attrs={"name": "dei:EntityRegistrantName"})
                if not entity_tag:
                    entity_tag = soup.find("ix:nonnumeric", attrs={"name": "dei:EntityRegistrantName"})

                if entity_tag and entity_tag.get_text(strip=True):
                    doc.company_name = entity_tag.get_text(strip=True)[:100]
                else:
                    # Try h1 heading near the top of the document
                    for tag in soup.find_all(["h1", "h2"])[:10]:
                        htext = tag.get_text(strip=True)
                        if htext and len(htext) < 100 and "apple" in htext.lower():
                            doc.company_name = htext
                            break
                    else:
                        # Fall back to title tag
                        title_tag = soup.find("title")
                        if title_tag and title_tag.get_text(strip=True):
                            doc.company_name = title_tag.get_text(strip=True)[:80]

                # Fix filing year: the _extract_year regex looks in first 5000 chars which is
                # usually XBRL preamble; also try matching the fiscal year from the filename or
                # document period of report element
                if not doc.filing_year or doc.filing_year == "1934":
                    period_tag = soup.find(attrs={"name": "dei:DocumentPeriodEndDate"})
                    if not period_tag:
                        period_tag = soup.find("ix:nonfraction", attrs={"name": "dei:DocumentPeriodEndDate"})
                    if not period_tag:
                        period_tag = soup.find("ix:nonnumeric", attrs={"name": "dei:DocumentPeriodEndDate"})

                    if period_tag:
                        period_text = period_tag.get_text(strip=True)
                        yr_match = _re.search(r"(\d{4})", period_text)
                        if yr_match:
                            doc.filing_year = yr_match.group(1)
                    else:
                        # Extract year from filename (e.g. aapl-20250927.htm → 2025)
                        fn_match = _re.search(r"-(\d{4})\d{4}", pdf_path.stem)
                        if fn_match:
                            doc.filing_year = fn_match.group(1)
                        else:
                            yr_match = _re.search(r"(202\d)", full_text[:10000])
                            if yr_match:
                                doc.filing_year = yr_match.group(1)

                logger.success(
                    f"HTML extraction complete: {len(chunks)} chunks, "
                    f"{len(full_text):,} chars"
                )

            except Exception as exc:
                logger.error(f"Failed to parse HTML {pdf_path}: {exc}")
                doc.extraction_errors.append(str(exc))
        else:
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    doc.page_count = len(pdf.pages)
                    pages_to_process = pdf.pages[: self.max_pages]

                    # Extract metadata
                    meta = pdf.metadata or {}
                    doc.company_name = self._clean_str(
                        meta.get("Author") or meta.get("Creator") or ""
                    )

                    for page_num, page in enumerate(pages_to_process, start=1):
                        # Extract text
                        try:
                            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                            doc.pages.append(text)
                        except Exception as exc:
                            logger.warning(f"Text extraction failed on page {page_num}: {exc}")
                            doc.pages.append("")
                            doc.extraction_errors.append(f"Page {page_num}: {exc}")

                        # Extract tables
                        try:
                            raw_tables = page.extract_tables()
                            for raw_table in (raw_tables or []):
                                table = self._parse_table(raw_table, page_num)
                                if table:
                                    doc.tables.append(table)
                        except Exception as exc:
                            logger.debug(f"Table extraction failed on page {page_num}: {exc}")

                    doc.raw_text = "\n\n".join(doc.pages)
                    doc.sections = self._classify_sections(doc.pages)
                    doc.filing_year = self._extract_year(doc.raw_text)

            except Exception as exc:
                logger.error(f"Failed to open PDF {pdf_path}: {exc}")
                doc.extraction_errors.append(str(exc))

        logger.success(
            f"Extracted {doc.page_count} pages, "
            f"{len(doc.tables)} tables, "
            f"{len(doc.sections)} sections from {pdf_path.name}"
        )
        return doc

    def get_text_around_position(
        self,
        doc: ExtractedDocument,
        char_pos: int,
        window: int = 500,
    ) -> str:
        """
        Return a window of text around a character position in raw_text.
        Used by internal_checker.py to find tables near a claim.
        """
        start = max(0, char_pos - window)
        end = min(len(doc.raw_text), char_pos + window)
        return doc.raw_text[start:end]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_table(
        self,
        raw_table: list[list[Optional[str]]],
        page_number: int,
    ) -> Optional[TableData]:
        """Convert a raw pdfplumber table to TableData."""
        if not raw_table or len(raw_table) < 2:
            return None

        # First non-empty row as headers
        headers = [str(cell) if cell else "" for cell in raw_table[0]]
        rows = []
        for raw_row in raw_table[1:]:
            row = [str(cell) if cell else "" for cell in raw_row]
            # Skip rows that are entirely empty
            if any(c.strip() for c in row):
                rows.append(row)

        if not rows:
            return None

        return TableData(
            page_number=page_number,
            headers=headers,
            rows=rows,
            raw=raw_table,
        )

    def _classify_sections(self, pages: list[str]) -> dict[str, str]:
        """
        Heuristically split page text into E/S/G sections.

        Strategy: slide a window of pages, check if any heading-like line
        matches a section keyword, accumulate pages until next section starts.
        """
        sections: dict[str, list[str]] = {k: [] for k in _SECTION_PATTERNS}
        current_section = "general"

        for page_text in pages:
            # Check first 5 lines of the page for a section heading
            first_lines = page_text.split("\n")[:5]
            heading_text = " ".join(first_lines)

            # If this page starts a known financial section, reset to general
            # This prevents MD&A, results-of-operations etc. from polluting ESG sections
            if _FINANCIAL_SECTION_BLOCKLIST.search(heading_text):
                current_section = "general"
            else:
                detected = self._detect_section(heading_text)
                if detected:
                    current_section = detected

            sections[current_section].append(page_text)

        return {k: "\n\n".join(v) for k, v in sections.items() if v}

    def _detect_section(self, heading_text: str) -> Optional[str]:
        """Return section name if heading matches known patterns, else None."""
        for section_name, patterns in _SECTION_PATTERNS.items():
            if section_name == "general":
                continue
            for pattern in patterns:
                if re.search(pattern, heading_text, re.IGNORECASE):
                    return section_name
        return None

    def _extract_year(self, text: str) -> str:
        """Try to extract the fiscal year from the document text."""
        # Look for "fiscal year 2023" or "year ended December 31, 2023"
        patterns = [
            r"fiscal year (\d{4})",
            r"year ended (?:December|January|March|June|September) \d+,?\s*(\d{4})",
            r"annual report.*?(\d{4})",
            r"for the year (\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _clean_str(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
