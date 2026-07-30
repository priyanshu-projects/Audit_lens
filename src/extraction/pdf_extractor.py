from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pdfplumber
from loguru import logger

@dataclass
class TableData:
    page_number: int
    headers: list[str]
    rows: list[list[str]]
    raw: list[list[Optional[str]]] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [' | '.join((str(h) for h in self.headers))]
        lines += [' | '.join((str(c) if c else '' for c in row)) for row in self.rows]
        return '\n'.join(lines)

@dataclass
class ExtractedDocument:
    source_path: Path
    company_name: str = ''
    ticker: str = ''
    filing_year: str = ''
    raw_text: str = ''
    pages: list[str] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    page_count: int = 0
    extraction_errors: list[str] = field(default_factory=list)
_FINANCIAL_SECTION_BLOCKLIST = re.compile('\\b(?:management.{0,10}discussion|MD&A|results of operations|liquidity and capital|financial condition|net sales|gross margin|operating income|net income|revenue|segment information|geographic information|selected financial|consolidated statements|balance sheet|income statement|cash flow|risk factors|forward.looking|legal proceedings|market for registrant|quantitative.*disclosures about market risk)\\b', re.IGNORECASE)
_SECTION_PATTERNS: dict[str, list[str]] = {'environmental': ['\\benvironmental\\b', '\\bclimate change\\b', '\\bclimate risk\\b', '\\bemission(?:s)?\\b', '\\bcarbon\\b', '\\bclean energy\\b', '\\brenewable energy\\b', '\\benergy consumption\\b', '\\bwater\\b.{0,20}\\bsustainab\\b', '\\bwaste\\b.{0,20}\\bsustainab\\b', '\\bbiodiversity\\b', '\\bgreenhous\\b', '\\bscope [123]\\b', '\\bnet.?zero\\b', '\\bcarbon.?neutral\\b'], 'social': ['\\bhuman capital\\b', '\\bworkforce\\b.{0,30}\\bdivers\\b', '\\bdiversity.*equity.*inclusion\\b', '\\bDEI\\b', '\\bemployee.*safety\\b', '\\bworkplace.*safety\\b', '\\bhuman rights\\b', '\\blabor.*standard\\b', '\\bsupply chain.*labor\\b', '\\bcommunity.*invest\\b', '\\bsocial impact\\b', '\\bphilanthrop\\b', '\\bdata privacy\\b.{0,40}\\bcommit\\b'], 'governance': ['\\bcorporate governance\\b', '\\bboard of directors\\b', '\\baudit committee\\b', '\\bexecutive compensation\\b', '\\banti.corruption\\b', '\\banti.bribery\\b', '\\bwhistleblower\\b', '\\bcode of (?:ethics|conduct)\\b', '\\bGRI\\b.{0,30}\\bindex\\b', '\\bsustainability.*governance\\b', '\\btransparency.*report\\b'], 'general': []}

class PdfExtractor:
    MAX_PAGES = 200

    def __init__(self, max_pages: int=MAX_PAGES) -> None:
        self.max_pages = max_pages

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        logger.info(f'Extracting: {pdf_path.name}')
        doc = ExtractedDocument(source_path=pdf_path)
        if not pdf_path.exists():
            doc.extraction_errors.append(f'File not found: {pdf_path}')
            logger.error(f'PDF not found: {pdf_path}')
            return doc
        is_html = pdf_path.suffix.lower() in ('.htm', '.html')
        if is_html:
            try:
                from bs4 import BeautifulSoup
                import re as _re
                logger.info(f'Extracting HTML/iXBRL filing: {pdf_path.name}')
                raw_html = pdf_path.read_text(encoding='utf-8', errors='replace')
                try:
                    soup = BeautifulSoup(raw_html, 'lxml')
                except Exception:
                    soup = BeautifulSoup(raw_html, 'html.parser')
                for tag in soup(['script', 'style', 'ix:hidden', 'ix:header', 'link', 'meta']):
                    tag.decompose()
                full_text = soup.get_text(separator='\n', strip=True)
                full_text = _re.sub('\\n{3,}', '\n\n', full_text)
                chunk_size = 3000
                chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
                chunks = chunks[:self.max_pages]
                doc.pages = chunks
                doc.page_count = len(chunks)
                doc.raw_text = full_text
                doc.sections = self._classify_sections(chunks)
                doc.filing_year = self._extract_year(full_text)
                entity_tag = soup.find(attrs={'name': 'dei:EntityRegistrantName'})
                if not entity_tag:
                    entity_tag = soup.find('ix:nonfraction', attrs={'name': 'dei:EntityRegistrantName'})
                if not entity_tag:
                    entity_tag = soup.find('ix:nonnumeric', attrs={'name': 'dei:EntityRegistrantName'})
                if entity_tag and entity_tag.get_text(strip=True):
                    doc.company_name = entity_tag.get_text(strip=True)[:100]
                else:
                    for tag in soup.find_all(['h1', 'h2'])[:10]:
                        htext = tag.get_text(strip=True)
                        if htext and len(htext) < 100 and ('apple' in htext.lower()):
                            doc.company_name = htext
                            break
                    else:
                        title_tag = soup.find('title')
                        if title_tag and title_tag.get_text(strip=True):
                            doc.company_name = title_tag.get_text(strip=True)[:80]
                if not doc.filing_year or doc.filing_year == '1934':
                    period_tag = soup.find(attrs={'name': 'dei:DocumentPeriodEndDate'})
                    if not period_tag:
                        period_tag = soup.find('ix:nonfraction', attrs={'name': 'dei:DocumentPeriodEndDate'})
                    if not period_tag:
                        period_tag = soup.find('ix:nonnumeric', attrs={'name': 'dei:DocumentPeriodEndDate'})
                    if period_tag:
                        period_text = period_tag.get_text(strip=True)
                        yr_match = _re.search('(\\d{4})', period_text)
                        if yr_match:
                            doc.filing_year = yr_match.group(1)
                    else:
                        fn_match = _re.search('-(\\d{4})\\d{4}', pdf_path.stem)
                        if fn_match:
                            doc.filing_year = fn_match.group(1)
                        else:
                            yr_match = _re.search('(202\\d)', full_text[:10000])
                            if yr_match:
                                doc.filing_year = yr_match.group(1)
                logger.success(f'HTML extraction complete: {len(chunks)} chunks, {len(full_text):,} chars')
            except Exception as exc:
                logger.error(f'Failed to parse HTML {pdf_path}: {exc}')
                doc.extraction_errors.append(str(exc))
        else:
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    doc.page_count = len(pdf.pages)
                    pages_to_process = pdf.pages[:self.max_pages]
                    meta = pdf.metadata or {}
                    doc.company_name = self._clean_str(meta.get('Author') or meta.get('Creator') or '')
                    for page_num, page in enumerate(pages_to_process, start=1):
                        try:
                            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ''
                            doc.pages.append(text)
                        except Exception as exc:
                            logger.warning(f'Text extraction failed on page {page_num}: {exc}')
                            doc.pages.append('')
                            doc.extraction_errors.append(f'Page {page_num}: {exc}')
                        try:
                            raw_tables = page.extract_tables()
                            for raw_table in raw_tables or []:
                                table = self._parse_table(raw_table, page_num)
                                if table:
                                    doc.tables.append(table)
                        except Exception as exc:
                            logger.debug(f'Table extraction failed on page {page_num}: {exc}')
                    doc.raw_text = '\n\n'.join(doc.pages)
                    doc.sections = self._classify_sections(doc.pages)
                    doc.filing_year = self._extract_year(doc.raw_text)
            except Exception as exc:
                logger.error(f'Failed to open PDF {pdf_path}: {exc}')
                doc.extraction_errors.append(str(exc))
        logger.success(f'Extracted {doc.page_count} pages, {len(doc.tables)} tables, {len(doc.sections)} sections from {pdf_path.name}')
        return doc

    def get_text_around_position(self, doc: ExtractedDocument, char_pos: int, window: int=500) -> str:
        start = max(0, char_pos - window)
        end = min(len(doc.raw_text), char_pos + window)
        return doc.raw_text[start:end]

    def _parse_table(self, raw_table: list[list[Optional[str]]], page_number: int) -> Optional[TableData]:
        if not raw_table or len(raw_table) < 2:
            return None
        headers = [str(cell) if cell else '' for cell in raw_table[0]]
        rows = []
        for raw_row in raw_table[1:]:
            row = [str(cell) if cell else '' for cell in raw_row]
            if any((c.strip() for c in row)):
                rows.append(row)
        if not rows:
            return None
        return TableData(page_number=page_number, headers=headers, rows=rows, raw=raw_table)

    def _classify_sections(self, pages: list[str]) -> dict[str, str]:
        sections: dict[str, list[str]] = {k: [] for k in _SECTION_PATTERNS}
        current_section = 'general'
        for page_text in pages:
            first_lines = page_text.split('\n')[:5]
            heading_text = ' '.join(first_lines)
            if _FINANCIAL_SECTION_BLOCKLIST.search(heading_text):
                current_section = 'general'
            else:
                detected = self._detect_section(heading_text)
                if detected:
                    current_section = detected
            sections[current_section].append(page_text)
        return {k: '\n\n'.join(v) for k, v in sections.items() if v}

    def _detect_section(self, heading_text: str) -> Optional[str]:
        for section_name, patterns in _SECTION_PATTERNS.items():
            if section_name == 'general':
                continue
            for pattern in patterns:
                if re.search(pattern, heading_text, re.IGNORECASE):
                    return section_name
        return None

    def _extract_year(self, text: str) -> str:
        patterns = ['fiscal year (\\d{4})', 'year ended (?:December|January|March|June|September) \\d+,?\\s*(\\d{4})', 'annual report.*?(\\d{4})', 'for the year (\\d{4})']
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return match.group(1)
        return ''

    @staticmethod
    def _clean_str(s: str) -> str:
        return re.sub('\\s+', ' ', s).strip()