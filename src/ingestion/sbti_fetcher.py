from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger
DEFAULT_CSV_PATH = Path('data/raw/sbti_targets.csv')

class SbtiFetcher:

    def __init__(self, csv_path: Path=DEFAULT_CSV_PATH) -> None:
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        if not self.csv_path.exists():
            logger.warning(f'SBTi data file not found at {self.csv_path}. SBTi checking will be skipped.')
            return
        try:
            self._df = pd.read_csv(self.csv_path)
            self._df['normalized_name'] = self._df['Company Name'].str.lower().str.replace('[^\\w\\s]', '', regex=True)
            logger.success(f'Loaded SBTi dataset: {len(self._df)} companies')
        except Exception as e:
            logger.error(f'Failed to load SBTi data: {e}')

    def get_company_data(self, company_name: str) -> Optional[dict]:
        if self._df is None or self._df.empty:
            return None
        search_name = company_name.lower().replace('[^\\w\\s]', '')
        match = self._df[self._df['normalized_name'] == search_name]
        if match.empty:
            first_word = search_name.split()[0]
            match = self._df[self._df['normalized_name'].str.contains(first_word, na=False)]
        if match.empty:
            return None
        row = match.iloc[0]
        return {'company_name': row['Company Name'], 'sector': row['Sector'], 'near_term_status': row['Near Term Target Status'], 'net_zero_status': row['Net-Zero Target Status'], 'net_zero_year': str(row['Net-Zero Target Year']).split('.')[0] if pd.notna(row['Net-Zero Target Year']) else None}