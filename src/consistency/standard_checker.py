from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from src.consistency.internal_checker import ConsistencyResult
from src.extraction.claim_detector import Claim
from src.rag.vector_store import VectorStore

@dataclass
class ComplianceRule:
    standard: str
    trigger_keywords: list[str]
    required_elements: list[str]
    description: str
COMPLIANCE_RULES: list[ComplianceRule] = [ComplianceRule(standard='GRI 305-1', trigger_keywords=['scope 1', 'direct emission', 'direct ghg'], required_elements=['metric tonnes CO2 equivalent', 'base year', 'consolidation approach (equity share or operational control)', 'biogenic CO2 exclusion note'], description='Direct (Scope 1) GHG emissions disclosure'), ComplianceRule(standard='GRI 305-2', trigger_keywords=['scope 2', 'indirect emission', 'purchased electricity'], required_elements=['location-based calculation', 'market-based calculation', 'base year'], description='Energy indirect (Scope 2) GHG emissions disclosure'), ComplianceRule(standard='GRI 305-3', trigger_keywords=['scope 3', 'value chain emission', 'upstream', 'downstream'], required_elements=['list of Scope 3 categories included', 'calculation methodology', 'base year'], description='Other indirect (Scope 3) GHG emissions disclosure'), ComplianceRule(standard='TCFD Physical Risk', trigger_keywords=['climate risk', 'physical risk', 'extreme weather', 'flood risk'], required_elements=['time horizon (short/medium/long term)', 'financial impact assessment', 'scenario analysis (1.5°C / 2°C)'], description='TCFD physical climate risk disclosure'), ComplianceRule(standard='TCFD Transition Risk', trigger_keywords=['transition risk', 'stranded asset', 'carbon price', 'regulatory risk'], required_elements=['time horizon', 'financial impact', 'scenario analysis'], description='TCFD transition risk disclosure'), ComplianceRule(standard='GRI 401-1', trigger_keywords=['employee', 'hire', 'turnover', 'workforce', 'new hire'], required_elements=['number of new hires', 'employee turnover rate', 'breakdown by age group', 'breakdown by gender', 'breakdown by region'], description='GRI 401-1 New employee hires and turnover'), ComplianceRule(standard='TCFD Net Zero', trigger_keywords=['net zero', 'net-zero', 'carbon neutral', 'carbon neutrality'], required_elements=['target year', 'baseline year', 'scope coverage (1, 2, 3)', 'interim targets', 'transition plan'], description='TCFD/SBTi net-zero target disclosure'), ComplianceRule(standard='GRI 302-1', trigger_keywords=['energy consumption', 'energy use', 'renewable energy', 'kwh', 'mwh', 'gwh'], required_elements=['energy consumed in joules or multiples', 'breakdown by fuel type', 'percentage from renewable sources'], description='GRI 302-1 Energy consumption within organization')]

class StandardChecker:

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def check(self, claim: Claim, top_k: int=3) -> ConsistencyResult:
        logger.info(f"L3 standard check for: '{claim.text[:60]}...'")
        claim_lower = claim.text.lower()
        matched_rule = self._match_rule(claim_lower)
        search_results = self.vector_store.search(claim.text, top_k=top_k)
        retrieved_standard = search_results[0].chunk.standard if search_results else 'UNKNOWN'
        retrieved_section = search_results[0].chunk.section if search_results else ''
        retrieval_confidence = search_results[0].score if search_results else 0.0
        if not matched_rule:
            return ConsistencyResult(level=3, status='PARTIALLY_SUPPORTED', note=f'No specific compliance rule matched for this claim. Closest standard retrieved via RAG: {retrieved_standard} — {retrieved_section}. Auditor should review against applicable framework manually.', extra={'retrieved_standard': retrieved_standard, 'retrieval_confidence': round(retrieval_confidence, 3)})
        missing, present = self._check_required_elements(claim.text, matched_rule)
        if not missing:
            return ConsistencyResult(level=3, status='SUPPORTED', note=f'Claim appears to meet the required disclosure elements for {matched_rule.standard} ({matched_rule.description}). All {len(present)} required elements found.', extra={'matched_standard': matched_rule.standard, 'present_elements': present, 'retrieved_standard': retrieved_standard})
        else:
            status = 'UNSUPPORTED' if len(missing) >= len(matched_rule.required_elements) // 2 else 'PARTIALLY_SUPPORTED'
            return ConsistencyResult(level=3, status=status, note=f"Claim relates to {matched_rule.standard} but is MISSING required disclosure elements. Present: {present or ['none']}. Missing: {missing}.", extra={'matched_standard': matched_rule.standard, 'missing_elements': missing, 'present_elements': present, 'retrieved_standard': retrieved_standard, 'retrieval_confidence': round(retrieval_confidence, 3)})

    @staticmethod
    def _match_rule(claim_lower: str) -> Optional[ComplianceRule]:
        for rule in COMPLIANCE_RULES:
            if any((kw in claim_lower for kw in rule.trigger_keywords)):
                return rule
        return None

    @staticmethod
    def _check_required_elements(claim_text: str, rule: ComplianceRule) -> tuple[list[str], list[str]]:
        claim_lower = claim_text.lower()
        missing = []
        present = []
        element_keywords: dict[str, list[str]] = {'metric tonnes CO2 equivalent': ['tco2', 'tonnes co2', 'mt co2', 'metric ton'], 'base year': ['base year', 'baseline year', 'since 20', 'from 20', '2019', '2020', '2021'], 'consolidation approach (equity share or operational control)': ['equity share', 'operational control', 'financial control'], 'biogenic CO2 exclusion note': ['biogenic', 'land use', 'lulucf'], 'location-based calculation': ['location-based', 'location based'], 'market-based calculation': ['market-based', 'market based', 'renewable energy certificate'], 'list of Scope 3 categories included': ['category', 'categories', 'upstream', 'downstream'], 'calculation methodology': ['methodology', 'method', 'calculation', 'protocol'], 'time horizon (short/medium/long term)': ['short-term', 'medium-term', 'long-term', '2030', '2040', '2050'], 'financial impact assessment': ['financial impact', '$ million', 'usd', 'material'], 'scenario analysis (1.5°C / 2°C)': ['scenario', '1.5', '2°c', '2 degree', 'ipcc'], 'target year': ['by 20', '2030', '2040', '2050', 'target year'], 'scope coverage (1, 2, 3)': ['scope 1', 'scope 2', 'scope 3'], 'interim targets': ['interim', 'milestone', '2025', '2027'], 'transition plan': ['transition plan', 'roadmap', 'pathway'], 'number of new hires': ['new hire', 'hired', 'recruitment'], 'employee turnover rate': ['turnover', 'attrition'], 'breakdown by age group': ['age', 'under 30', '30-50'], 'breakdown by gender': ['gender', 'women', 'female', 'male'], 'breakdown by region': ['region', 'country', 'geographic'], 'energy consumed in joules or multiples': ['kwh', 'mwh', 'gwh', 'gj', 'tj', 'joule'], 'breakdown by fuel type': ['natural gas', 'diesel', 'electricity', 'coal', 'fuel'], 'percentage from renewable sources': ['renewable', 'solar', 'wind', 'hydro'], 'scenario analysis': ['scenario', '1.5', '2 degree'], 'time horizon': ['2030', '2040', '2050', 'short', 'medium', 'long'], 'financial impact': ['financial', '$ million', 'usd', 'cost']}
        for element in rule.required_elements:
            keywords = element_keywords.get(element, [element.lower()])
            if any((kw in claim_lower for kw in keywords)):
                present.append(element)
            else:
                missing.append(element)
        return (missing, present)