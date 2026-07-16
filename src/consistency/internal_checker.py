"""
src/consistency/internal_checker.py
=====================================
Level 1 Consistency Check — Internal Document Consistency.

Question: "Is there a supporting data table in THIS document that
           backs up the quantitative claim?"

Example:
  Claim: "We reduced Scope 1 emissions by 40% since 2019."
  L1 Check: Does the same PDF contain a table with Scope 1 emission
             figures for 2019 and the current year? If yes → SUPPORTED.
             If no → UNSUPPORTED (raise flag for auditor).

This is the FASTEST check — pure text/table matching, no external calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from src.extraction.pdf_extractor import ExtractedDocument


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    """Result of a single consistency check level."""
    level: int                     # 1 | 2 | 3 | 4
    status: str                    # SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | SKIPPED
    evidence: str = ""             # text snippet that supports/contradicts the claim
    note: str = ""                 # human-readable explanation for auditor
    extra: dict = field(default_factory=dict)   # level-specific metadata


# ── Number extraction ─────────────────────────────────────────────────────────
# Regex to find numeric values in a claim — used to search for matching numbers in tables

_NUMBER_PATTERN = re.compile(
    r"""
    (?:                     # optional prefix: $, £, etc.
        [\$£€]?
    )
    \b
    (\d[\d,]*               # integer part
    (?:\.\d+)?)             # optional decimal
    \s*
    (?:%|percent|            # percentage
    tonne|tonnes|            # mass units
    tCO2|CO2|               # carbon
    MWh|GWh|kWh|            # energy
    gallon|litre|liter|      # volume
    MW|GW)?                  # power
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


class InternalChecker:
    """
    Level 1: Internal document consistency check.

    Searches the same PDF for:
    1. Numeric values from the claim appearing in nearby tables
    2. Year references from the claim appearing in data tables
    3. ESG-category keywords matching the claim's section
    """

    # How many characters around the claim position to search for supporting tables
    SEARCH_WINDOW_CHARS = 3000

    def check(
        self,
        claim_text: str,
        claim_start_char: int,
        extracted_doc: ExtractedDocument,
    ) -> ConsistencyResult:
        """
        Check if the document contains supporting numerical data for the claim.

        Args:
            claim_text:       The ESG claim sentence
            claim_start_char: Character position of the claim in raw_text
            extracted_doc:    Full extracted document (text + tables)

        Returns:
            ConsistencyResult with level=1
        """
        logger.debug(f"L1 internal check for: '{claim_text[:60]}...'")

        # 1. Extract numbers and years from the claim
        claim_numbers = self._extract_numbers(claim_text)
        claim_years = self._extract_years(claim_text)

        if not claim_numbers:
            # No quantitative content to verify
            return ConsistencyResult(
                level=1,
                status="SKIPPED",
                note=(
                    "No numeric values found in this claim — "
                    "internal consistency check not applicable."
                ),
            )

        # 2. Get text window around the claim (±SEARCH_WINDOW_CHARS)
        start = max(0, claim_start_char - self.SEARCH_WINDOW_CHARS)
        end = min(len(extracted_doc.raw_text), claim_start_char + self.SEARCH_WINDOW_CHARS)
        context_window = extracted_doc.raw_text[start:end]

        # 3. Check tables for matching numbers
        table_matches = self._check_tables(claim_numbers, claim_years, extracted_doc.tables)

        # 4. Check surrounding text for matching numbers
        text_matches = self._check_text_window(claim_numbers, claim_years, context_window)

        # 5. Determine status
        if table_matches:
            return ConsistencyResult(
                level=1,
                status="SUPPORTED",
                evidence=table_matches[0]["evidence"],
                note=(
                    f"Found supporting data table in document "
                    f"(page {table_matches[0]['page']}). "
                    f"Numeric value '{table_matches[0]['matched_value']}' verified."
                ),
                extra={"table_matches": table_matches},
            )
        elif text_matches:
            return ConsistencyResult(
                level=1,
                status="PARTIALLY_SUPPORTED",
                evidence=text_matches[0]["context"],
                note=(
                    "Matching numeric value found in document text "
                    "but not in a formal data table. "
                    "Auditor should verify if the data is from a structured disclosure."
                ),
                extra={"text_matches": text_matches},
            )
        else:
            return ConsistencyResult(
                level=1,
                status="UNSUPPORTED",
                note=(
                    f"Could not find supporting data table for the claimed "
                    f"value(s) {claim_numbers} in the document. "
                    "This claim lacks internal numerical evidence."
                ),
                extra={"searched_numbers": claim_numbers, "searched_years": claim_years},
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        """Extract numeric values from a claim sentence."""
        matches = _NUMBER_PATTERN.findall(text)
        # Clean up and filter very small numbers (< 0.1) and pure counts
        cleaned = []
        text_lower = text.lower()
        for m in matches:
            num_str = m.replace(",", "")
            try:
                val = float(num_str)
                if val >= 0.1:
                    # Ignore number if it is part of "Scope 1", "Scope 2", "Scope 3"
                    if f"scope {num_str}" in text_lower:
                        continue
                    cleaned.append(m)
            except ValueError:
                pass
        return cleaned[:5]   # top 5 numbers

    @staticmethod
    def _extract_years(text: str) -> list[str]:
        """Extract 4-digit year references from a claim."""
        return _YEAR_PATTERN.findall(text)

    @staticmethod
    def _check_tables(
        claim_numbers: list[str],
        claim_years: list[str],
        tables: list,
    ) -> list[dict]:
        """
        Search extracted tables for numeric values matching the claim.
        Returns list of match dicts with evidence.
        """
        matches = []
        for table in tables:
            table_text = table.to_text()

            for num in claim_numbers:
                # Look for the number in table text (allow for formatting differences)
                num_clean = num.replace(",", "")
                if num_clean in table_text.replace(",", ""):
                    matches.append({
                        "matched_value": num,
                        "page": table.page_number,
                        "evidence": table_text[:300],
                    })

            # Also check for year columns in table headers
            if claim_years:
                header_text = " ".join(str(h) for h in table.headers)
                for year in claim_years:
                    if year in header_text:
                        if not matches:  # partial match if no number match yet
                            matches.append({
                                "matched_value": f"year {year}",
                                "page": table.page_number,
                                "evidence": header_text[:200],
                            })

        return matches

    @staticmethod
    def _check_text_window(
        claim_numbers: list[str],
        claim_years: list[str],
        context_window: str,
    ) -> list[dict]:
        """Search surrounding text for numeric values matching the claim."""
        matches = []
        for num in claim_numbers:
            num_clean = num.replace(",", "")
            # Find position of match in context
            idx = context_window.replace(",", "").find(num_clean)
            if idx >= 0:
                snippet_start = max(0, idx - 100)
                snippet_end = min(len(context_window), idx + 150)
                matches.append({
                    "matched_value": num,
                    "context": context_window[snippet_start:snippet_end],
                })
        return matches
