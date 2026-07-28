from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from src.consistency.internal_checker import ConsistencyResult, _NUMBER_PATTERN
from src.extraction.claim_detector import Claim
from src.extraction.pdf_extractor import PdfExtractor
from src.ingestion.edgar_fetcher import EdgarFetcher

class HistoricalChecker:

    def __init__(self, fetcher: Optional[EdgarFetcher]=None, extractor: Optional[PdfExtractor]=None) -> None:
        self.fetcher = fetcher or EdgarFetcher()
        self.extractor = extractor or PdfExtractor()
        self._prior_year_cache: dict[str, str] = {}

    def check(self, claim: Claim, ticker: str, current_year: int, output_dir=None) -> ConsistencyResult:
        from pathlib import Path
        output_dir = output_dir or Path('data/raw')
        if current_year == 0 or not ticker:
            return ConsistencyResult(level=2, status='SKIPPED', note='Filing year or ticker unknown — historical check skipped.')
        prior_year = current_year - 1
        logger.info(f'L2 historical check | {ticker} | {current_year} vs {prior_year}')
        cache_key = f'{ticker}_{prior_year}'
        if cache_key not in self._prior_year_cache:
            prior_text = self._fetch_prior_year_text(ticker, prior_year, output_dir)
            self._prior_year_cache[cache_key] = prior_text
        else:
            prior_text = self._prior_year_cache[cache_key]
        if not prior_text:
            return ConsistencyResult(level=2, status='SKIPPED', note=f'Prior year filing ({prior_year}) not available for {ticker}.')
        current_numbers = self._extract_numbers(claim.text)
        if not current_numbers:
            return ConsistencyResult(level=2, status='SKIPPED', note='No quantitative values in claim to cross-reference.')
        matches = self._find_related_sentences(claim.text, prior_text)
        if not matches:
            return ConsistencyResult(level=2, status='UNSUPPORTED', note=f'Could not find related disclosure in prior year ({prior_year}) filing. This may indicate a new disclosure or changed reporting scope.', extra={'current_numbers': current_numbers, 'prior_year': prior_year})
        best_match = matches[0]
        prior_numbers = self._extract_numbers(best_match)
        comparison = self._compare_numbers(current_numbers, prior_numbers)
        if comparison['delta_pct'] is None:
            return ConsistencyResult(level=2, status='PARTIALLY_SUPPORTED', evidence=best_match, note=f'Related disclosure found in {prior_year} filing but numeric values could not be directly compared. Auditor should review both disclosures manually.', extra={'prior_year': prior_year, 'prior_sentence': best_match, 'prior_numbers': prior_numbers})
        delta = comparison['delta_pct']
        delta_str = f'{delta:+.1f}%'
        if abs(delta) > 20:
            status = 'UNSUPPORTED'
            note = f'Significant year-over-year change detected ({delta_str}) — value changed from {prior_numbers[0]} ({prior_year}) to {current_numbers[0]} ({current_year}). Auditor should request explanation for this material change.'
        elif abs(delta) > 5:
            status = 'PARTIALLY_SUPPORTED'
            note = f'Moderate year-over-year change ({delta_str}). Prior year value: {prior_numbers[0]}, current year: {current_numbers[0]}. Verify that methodology is consistent and change is disclosed.'
        else:
            status = 'SUPPORTED'
            note = f'Consistent with prior year filing ({prior_year}). Year-over-year change: {delta_str} — within expected range.'
        return ConsistencyResult(level=2, status=status, evidence=best_match, note=note, extra={'prior_year': prior_year, 'delta_pct': delta, 'current_numbers': current_numbers, 'prior_numbers': prior_numbers})

    def _fetch_prior_year_text(self, ticker: str, year: int, output_dir) -> str:
        from pathlib import Path
        filing = self.fetcher.get_historical_filing(ticker, year)
        if not filing:
            return ''
        downloaded = self.fetcher.download_filing(filing, output_dir=Path(output_dir))
        if not downloaded.download_success:
            return ''
        if downloaded.local_path.suffix.lower() in ['.pdf']:
            doc = self.extractor.extract(downloaded.local_path)
            return doc.raw_text
        else:
            try:
                with open(downloaded.local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_html = f.read()
                clean = re.sub('<[^>]+>', ' ', raw_html)
                clean = re.sub('\\s+', ' ', clean)
                return clean[:100000]
            except Exception as exc:
                logger.error(f'Could not read HTML filing: {exc}')
                return ''

    @staticmethod
    def _find_related_sentences(claim_text: str, prior_text: str, top_n: int=3) -> list[str]:
        claim_words = set((w.lower() for w in re.findall('\\b[a-zA-Z]{4,}\\b', claim_text) if w.lower() not in {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'our'}))
        sentences = re.split('(?<=[.!?])\\s+', prior_text)
        scored = []
        for sent in sentences:
            if len(sent) < 30 or len(sent) > 500:
                continue
            sent_words = set((w.lower() for w in re.findall('\\b[a-zA-Z]{4,}\\b', sent)))
            overlap = len(claim_words & sent_words)
            if overlap >= 2:
                scored.append((overlap, sent.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_n]]

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        matches = _NUMBER_PATTERN.findall(text)
        return [m.replace(',', '') for m in matches if m]

    @staticmethod
    def _compare_numbers(current: list[str], prior: list[str]) -> dict:
        if not current or not prior:
            return {'delta_pct': None}
        try:
            curr_val = float(current[0])
            prior_val = float(prior[0])
            if prior_val == 0:
                return {'delta_pct': None}
            delta = (curr_val - prior_val) / abs(prior_val) * 100
            return {'delta_pct': round(delta, 2)}
        except (ValueError, IndexError):
            return {'delta_pct': None}