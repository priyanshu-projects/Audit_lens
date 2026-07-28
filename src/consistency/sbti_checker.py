from __future__ import annotations
import re
from typing import Optional
from loguru import logger
from src.consistency.internal_checker import ConsistencyResult
from src.extraction.claim_detector import Claim
from src.ingestion.sbti_fetcher import SbtiFetcher

class SbtiChecker:

    def __init__(self, fetcher: Optional[SbtiFetcher]=None) -> None:
        self.fetcher = fetcher or SbtiFetcher()

    def check(self, claim: Claim, company_name: str) -> Optional[ConsistencyResult]:
        if claim.claim_type != 'commitment':
            return ConsistencyResult(level=3, status='SKIPPED', note='L3 SBTi check only applies to commitment-type claims.')
        sbti_data = self.fetcher.get_company_data(company_name)
        if not sbti_data:
            logger.warning(f'L3 SBTi Check: {company_name} not found in SBTi database.')
            return ConsistencyResult(level=3, status='UNSUPPORTED', note=f'{company_name} is not listed in the SBTi database of verified targets. The claim cannot be cross-verified against external science-based targets.')
        year_match = re.search('\\b(20\\d{2})\\b', claim.text)
        claimed_year = year_match.group(1) if year_match else None
        if 'net zero' in claim.text.lower() or 'carbon neutral' in claim.text.lower():
            sbti_year = sbti_data.get('net_zero_year')
            sbti_status = sbti_data.get('net_zero_status')
            if sbti_status != 'Targets Set':
                return ConsistencyResult(level=3, status='HIGH_RISK', note=f"Company claims net-zero target, but SBTi status is '{sbti_status}'. This is a potential greenwashing signal.", evidence=f"SBTi DB: {sbti_data['company_name']} - {sbti_status}")
            if claimed_year and sbti_year and (claimed_year != sbti_year):
                return ConsistencyResult(level=3, status='HIGH_RISK', note=f'Claimed net-zero year ({claimed_year}) conflicts with SBTi verified year ({sbti_year}). This discrepancy requires auditor review.', evidence=f"SBTi DB: {sbti_data['company_name']} - Verified Net-Zero Year: {sbti_year}")
            return ConsistencyResult(level=3, status='SUPPORTED', note='Net-zero target aligns with verified SBTi data.', evidence=f"SBTi DB: {sbti_data['company_name']} - Status: {sbti_status} ({sbti_year or 'N/A'})")
        if sbti_data.get('near_term_status') == 'Targets Set':
            return ConsistencyResult(level=3, status='SUPPORTED', note='Company has verified near-term science-based targets.', evidence=f"SBTi DB: {sbti_data['company_name']} - Near-Term: Targets Set")
        return ConsistencyResult(level=3, status='PARTIALLY_SUPPORTED', note=f"Company has no verified near-term targets set in SBTi (status: {sbti_data.get('near_term_status')}).", evidence=f"SBTi DB: Near-Term Status = {sbti_data.get('near_term_status')}")