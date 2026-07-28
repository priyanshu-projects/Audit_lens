from __future__ import annotations
import re
from dataclasses import dataclass, field
from loguru import logger
from src.extraction.pdf_extractor import ExtractedDocument

@dataclass
class ConsistencyResult:
    level: int
    status: str
    evidence: str = ''
    note: str = ''
    extra: dict = field(default_factory=dict)
_NUMBER_PATTERN = re.compile('\n    (?:                     # optional prefix: $, £, etc.\n        [\\$£€]?\n    )\n    \\b\n    (\\d[\\d,]*               # integer part\n    (?:\\.\\d+)?)             # optional decimal\n    \\s*\n    (?:%|percent|            # percentage\n    tonne|tonnes|            # mass units\n    tCO2|CO2|               # carbon\n    MWh|GWh|kWh|            # energy\n    gallon|litre|liter|      # volume\n    MW|GW)?                  # power\n    \\b\n    ', re.VERBOSE | re.IGNORECASE)
_YEAR_PATTERN = re.compile('\\b(19|20)\\d{2}\\b')

class InternalChecker:
    SEARCH_WINDOW_CHARS = 3000

    def check(self, claim_text: str, claim_start_char: int, extracted_doc: ExtractedDocument) -> ConsistencyResult:
        logger.debug(f"L1 internal check for: '{claim_text[:60]}...'")
        claim_numbers = self._extract_numbers(claim_text)
        claim_years = self._extract_years(claim_text)
        if not claim_numbers:
            return ConsistencyResult(level=1, status='SKIPPED', note='No numeric values found in this claim — internal consistency check not applicable.')
        start = max(0, claim_start_char - self.SEARCH_WINDOW_CHARS)
        end = min(len(extracted_doc.raw_text), claim_start_char + self.SEARCH_WINDOW_CHARS)
        context_window = extracted_doc.raw_text[start:end]
        table_matches = self._check_tables(claim_numbers, claim_years, extracted_doc.tables)
        text_matches = self._check_text_window(claim_numbers, claim_years, context_window)
        if table_matches:
            return ConsistencyResult(level=1, status='SUPPORTED', evidence=table_matches[0]['evidence'], note=f"Found supporting data table in document (page {table_matches[0]['page']}). Numeric value '{table_matches[0]['matched_value']}' verified.", extra={'table_matches': table_matches})
        elif text_matches:
            return ConsistencyResult(level=1, status='PARTIALLY_SUPPORTED', evidence=text_matches[0]['context'], note='Matching numeric value found in document text but not in a formal data table. Auditor should verify if the data is from a structured disclosure.', extra={'text_matches': text_matches})
        else:
            return ConsistencyResult(level=1, status='UNSUPPORTED', note=f'Could not find supporting data table for the claimed value(s) {claim_numbers} in the document. This claim lacks internal numerical evidence.', extra={'searched_numbers': claim_numbers, 'searched_years': claim_years})

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        matches = _NUMBER_PATTERN.findall(text)
        cleaned = []
        text_lower = text.lower()
        for m in matches:
            num_str = m.replace(',', '')
            try:
                val = float(num_str)
                if val >= 0.1:
                    if f'scope {num_str}' in text_lower:
                        continue
                    cleaned.append(m)
            except ValueError:
                pass
        return cleaned[:5]

    @staticmethod
    def _extract_years(text: str) -> list[str]:
        return _YEAR_PATTERN.findall(text)

    @staticmethod
    def _check_tables(claim_numbers: list[str], claim_years: list[str], tables: list) -> list[dict]:
        matches = []
        for table in tables:
            table_text = table.to_text()
            for num in claim_numbers:
                num_clean = num.replace(',', '')
                if num_clean in table_text.replace(',', ''):
                    matches.append({'matched_value': num, 'page': table.page_number, 'evidence': table_text[:300]})
            if claim_years:
                header_text = ' '.join((str(h) for h in table.headers))
                for year in claim_years:
                    if year in header_text:
                        if not matches:
                            matches.append({'matched_value': f'year {year}', 'page': table.page_number, 'evidence': header_text[:200]})
        return matches

    @staticmethod
    def _check_text_window(claim_numbers: list[str], claim_years: list[str], context_window: str) -> list[dict]:
        matches = []
        for num in claim_numbers:
            num_clean = num.replace(',', '')
            idx = context_window.replace(',', '').find(num_clean)
            if idx >= 0:
                snippet_start = max(0, idx - 100)
                snippet_end = min(len(context_window), idx + 150)
                matches.append({'matched_value': num, 'context': context_window[snippet_start:snippet_end]})
        return matches