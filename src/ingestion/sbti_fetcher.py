"""
src/ingestion/sbti_fetcher.py
==============================
Fetches and caches the Science Based Targets initiative (SBTi) target dashboard.

SBTi provides a public dataset of companies and their verified climate targets.
This module provides lightning-fast lookups against that dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# For the demo, we use a cached CSV to avoid downloading the massive 20MB 
# SBTi excel file on every app start, but in production this could hit their API/Excel directly.
DEFAULT_CSV_PATH = Path("data/raw/sbti_targets.csv")


class SbtiFetcher:
    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH) -> None:
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        if not self.csv_path.exists():
            logger.warning(f"SBTi data file not found at {self.csv_path}. SBTi checking will be skipped.")
            return

        try:
            self._df = pd.read_csv(self.csv_path)
            # Normalize company names for robust lookups
            self._df["normalized_name"] = self._df["Company Name"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
            logger.success(f"Loaded SBTi dataset: {len(self._df)} companies")
        except Exception as e:
            logger.error(f"Failed to load SBTi data: {e}")

    def get_company_data(self, company_name: str) -> Optional[dict]:
        """
        Look up a company in the SBTi database.
        
        Args:
            company_name: The company name (e.g. "Apple Inc.")
            
        Returns:
            Dict of their target status, or None if not found.
        """
        if self._df is None or self._df.empty:
            return None

        # Normalize search string
        search_name = company_name.lower().replace(r"[^\w\s]", "")
        
        # Try exact match first
        match = self._df[self._df["normalized_name"] == search_name]
        
        # If no exact match, try substring (e.g. "Apple" matches "Apple Inc.")
        if match.empty:
            # We split the search name to just its first word to handle "Apple Inc." -> "Apple"
            first_word = search_name.split()[0]
            match = self._df[self._df["normalized_name"].str.contains(first_word, na=False)]

        if match.empty:
            return None
            
        # Return the first match
        row = match.iloc[0]
        return {
            "company_name": row["Company Name"],
            "sector": row["Sector"],
            "near_term_status": row["Near Term Target Status"],
            "net_zero_status": row["Net-Zero Target Status"],
            "net_zero_year": str(row["Net-Zero Target Year"]).split(".")[0] if pd.notna(row["Net-Zero Target Year"]) else None
        }
