from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from src.consistency.internal_checker import ConsistencyResult
STATUS_SCORES = {'SUPPORTED': 0.0, 'PARTIALLY_SUPPORTED': 0.4, 'UNSUPPORTED': 0.8, 'SKIPPED': 0.0, 'HIGH_RISK': 1.0}
LEVEL_WEIGHTS = {1: 0.25, 2: 0.35, 3: 0.4, 4: 0.5}

@dataclass
class AggregateResult:
    verdict: str
    risk_score: float
    summary: str
    level_results: dict[int, ConsistencyResult] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'verdict': self.verdict, 'risk_score': round(self.risk_score, 4), 'summary': self.summary, 'flags': self.flags, 'levels': {lvl: {'status': r.status, 'note': r.note} for lvl, r in self.level_results.items()}}
VERDICT_COLORS = {'CONSISTENT': '#22c55e', 'PARTIALLY_CONSISTENT': '#f59e0b', 'INCONSISTENT': '#ef4444', 'HIGH_RISK': '#7c3aed'}
VERDICT_EMOJI = {'CONSISTENT': '✅', 'PARTIALLY_CONSISTENT': '⚠️', 'INCONSISTENT': '❌', 'HIGH_RISK': '🚨'}
_NLP_FLAG_FALLBACK: dict[str, tuple[float, str]] = {'HIGH_RISK': (0.75, 'INCONSISTENT'), 'NEEDS_REVIEW': (0.4, 'PARTIALLY_CONSISTENT'), 'LIKELY_CONSISTENT': (0.15, 'CONSISTENT')}

def aggregate_results(l1: Optional[ConsistencyResult]=None, l2: Optional[ConsistencyResult]=None, l3: Optional[ConsistencyResult]=None, l4: Optional[ConsistencyResult]=None, nlp_flag: str='', esg_label: str='MIXED', nlp_confidence: float=0.0) -> AggregateResult:
    level_results: dict[int, ConsistencyResult] = {}
    flags: list[str] = []
    level_map = {1: l1, 2: l2, 3: l3, 4: l4}
    for lvl, result in level_map.items():
        if result is not None:
            level_results[lvl] = result
    if esg_label == 'MIXED' and nlp_confidence >= 0.8:
        return AggregateResult(verdict='NOT_ESG_CLAIM', risk_score=0.05, summary=f'FinBERT ESG model determined this sentence is not an ESG claim (confidence: {nlp_confidence:.0%}). Likely corporate boilerplate.', level_results=level_results, flags=['Not classified as E/S/G by fine-tuned ESG model.'])
    if not level_results:
        return AggregateResult(verdict='PARTIALLY_CONSISTENT', risk_score=0.5, summary='No consistency checks were run.', level_results={})
    total_weight = 0.0
    weighted_score = 0.0
    for lvl, result in level_results.items():
        if result.status == 'SKIPPED':
            continue
        weight = LEVEL_WEIGHTS.get(lvl, 0.2)
        score = STATUS_SCORES.get(result.status, 0.5)
        weighted_score += weight * score
        total_weight += weight
        if result.status == 'UNSUPPORTED':
            flags.append(f'L{lvl}: {result.note[:100]}')
    if l3 and l3.status == 'HIGH_RISK':
        return AggregateResult(verdict='HIGH_RISK', risk_score=1.0, summary='External verification discrepancy detected (SBTi conflict).', level_results=level_results, flags=flags + [f'L3 SBTi: {l3.note[:150]}'])
    if l4 and l4.status == 'UNSUPPORTED':
        return AggregateResult(verdict='HIGH_RISK', risk_score=1.0, summary='CDP submission discrepancy detected.', level_results=level_results, flags=flags + [f'L4 CDP: {l4.note[:150]}'])
    if total_weight == 0.0:
        fallback_risk, fallback_verdict = _NLP_FLAG_FALLBACK.get(nlp_flag, (0.3, 'PARTIALLY_CONSISTENT'))
        return AggregateResult(verdict=fallback_verdict, risk_score=round(fallback_risk, 4), summary=f"Structural consistency checks could not run for this claim (no numeric evidence to verify). Risk estimate is based on FinBERT NLP analysis: {nlp_flag or 'unknown'}.", level_results=level_results, flags=flags)
    risk_score = weighted_score / max(total_weight, 0.01)
    if risk_score < 0.2:
        verdict = 'CONSISTENT'
        summary = 'Claim is well-supported across all available evidence sources.'
    elif risk_score < 0.5:
        verdict = 'PARTIALLY_CONSISTENT'
        summary = 'Claim has partial support — some disclosure gaps identified.'
    elif risk_score < 0.8:
        verdict = 'INCONSISTENT'
        summary = 'Claim has significant inconsistencies — auditor review required.'
    else:
        verdict = 'HIGH_RISK'
        summary = 'Multiple consistency failures — potential greenwashing risk.'
    return AggregateResult(verdict=verdict, risk_score=round(risk_score, 4), summary=summary, level_results=level_results, flags=flags)