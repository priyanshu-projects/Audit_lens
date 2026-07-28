from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from loguru import logger

@dataclass
class TokenAttribution:
    token: str
    value: float
    position: int

@dataclass
class ShapResult:
    claim_text: str
    token_attributions: list[TokenAttribution] = field(default_factory=list)
    top_risk_tokens: list[TokenAttribution] = field(default_factory=list)
    top_safe_tokens: list[TokenAttribution] = field(default_factory=list)
    narrative: str = ''
    base_value: float = 0.0
    prediction_label: str = ''

    def to_dict(self) -> dict:
        return {'claim_text': self.claim_text, 'narrative': self.narrative, 'top_risk_tokens': [{'token': t.token, 'value': round(t.value, 4)} for t in self.top_risk_tokens], 'top_safe_tokens': [{'token': t.token, 'value': round(t.value, 4)} for t in self.top_safe_tokens], 'base_value': round(self.base_value, 4), 'prediction_label': self.prediction_label}
_NARRATIVE_TEMPLATES = {'quantified_no_baseline': 'Quantified {esg_type} claim detected. Risk is elevated because no baseline year is cited — auditor should ask: compared to what?', 'commitment_no_plan': 'Forward commitment claim detected. Risk is elevated because no implementation plan or interim target is referenced.', 'generic': 'This claim was flagged because the following words raised the risk score: {risk_tokens}. The following words lowered the risk score: {safe_tokens}.'}

class ShapExplainer:
    TOP_K = 5

    def __init__(self, classifier=None, tokenizer=None) -> None:
        self._classifier = classifier
        self._tokenizer = tokenizer
        self._shap_explainer = None
        self._cache: dict[str, ShapResult] = {}
        logger.info('ShapExplainer initialised (lazy — explainer loads on first .explain() call)')

    def explain(self, claim_text: str, prediction_label: str='') -> ShapResult:
        cache_key = claim_text.strip()
        if cache_key in self._cache:
            logger.debug(f'SHAP cache hit for: {claim_text[:50]}...')
            return self._cache[cache_key]
        logger.info(f'Computing SHAP for: {claim_text[:60]}...')
        try:
            shap_values, base_value = self._compute_shap(claim_text)
            attributions = self._build_attributions(claim_text, shap_values)
        except Exception as exc:
            logger.warning(f'SHAP computation failed: {exc}. Using heuristic fallback.')
            attributions, base_value = self._heuristic_attributions(claim_text)
        sorted_asc = sorted(attributions, key=lambda t: t.value)
        sorted_desc = sorted(attributions, key=lambda t: t.value, reverse=True)
        top_risk = [t for t in sorted_desc if t.value > 0][:self.TOP_K]
        top_safe = [t for t in sorted_asc if t.value < 0][:self.TOP_K]
        narrative = self._generate_narrative(claim_text, top_risk, top_safe, prediction_label)
        result = ShapResult(claim_text=claim_text, token_attributions=attributions, top_risk_tokens=top_risk, top_safe_tokens=top_safe, narrative=narrative, base_value=base_value, prediction_label=prediction_label)
        self._cache[cache_key] = result
        return result

    def _get_shap_explainer(self):
        if self._shap_explainer is not None:
            return self._shap_explainer
        import shap
        from transformers import pipeline
        if self._classifier is not None:
            pipe = pipeline('text-classification', model=self._classifier.model, tokenizer=self._classifier.tokenizer, device=-1, return_all_scores=True)
        else:
            from config.settings import hf_cfg
            pipe = pipeline('text-classification', model=hf_cfg.finbert_model_id, device=-1, return_all_scores=True)
        self._shap_explainer = shap.Explainer(pipe)
        logger.success('SHAP Explainer initialised')
        return self._shap_explainer

    def _compute_shap(self, claim_text: str) -> tuple[np.ndarray, float]:
        explainer = self._get_shap_explainer()
        shap_values = explainer([claim_text])
        values = shap_values.values[0].sum(axis=-1)
        base = float(shap_values.base_values[0].mean())
        return (values, base)

    def _build_attributions(self, claim_text: str, shap_values: np.ndarray) -> list[TokenAttribution]:
        if self._classifier is not None:
            tokenizer = self._classifier.tokenizer
        else:
            from transformers import AutoTokenizer
            from config.settings import hf_cfg
            tokenizer = AutoTokenizer.from_pretrained(hf_cfg.finbert_model_id)
        tokens = tokenizer.tokenize(claim_text)
        n = min(len(tokens), len(shap_values))
        attributions = []
        for i in range(n):
            tok = tokens[i].replace('##', '')
            val = float(shap_values[i])
            attributions.append(TokenAttribution(token=tok, value=val, position=i))
        return attributions

    def _heuristic_attributions(self, claim_text: str) -> tuple[list[TokenAttribution], float]:
        RISK_WORDS = {'target': 0.15, 'commit': 0.12, 'plan': 0.1, 'aspire': 0.13, 'goal': 0.11, 'since': 0.14, 'baseline': 0.16, 'expect': 0.09}
        SAFE_WORDS = {'reduced': -0.12, 'achieved': -0.14, 'verified': -0.18, 'certified': -0.16, 'completed': -0.13, 'third-party': -0.2, '%': -0.08, 'tonne': -0.1, 'kwh': -0.09}
        attributions = []
        words = claim_text.lower().split()
        for i, word in enumerate(words):
            val = 0.0
            clean = word.strip('.,;:"\'()[]')
            if clean in RISK_WORDS:
                val = RISK_WORDS[clean]
            elif clean in SAFE_WORDS:
                val = SAFE_WORDS[clean]
            attributions.append(TokenAttribution(token=word, value=val, position=i))
        return (attributions, 0.0)

    def _generate_narrative(self, claim_text: str, top_risk: list[TokenAttribution], top_safe: list[TokenAttribution], prediction_label: str) -> str:
        claim_lower = claim_text.lower()
        risk_token_strs = ', '.join((f"'{t.token}'" for t in top_risk)) if top_risk else 'none'
        safe_token_strs = ', '.join((f"'{t.token}'" for t in top_safe)) if top_safe else 'none'
        has_percent = '%' in claim_text
        has_year = any((str(y) in claim_text for y in range(1990, 2040)))
        has_baseline = any((w in claim_lower for w in ['since', 'baseline', 'compared', 'versus']))
        has_commitment = any((w in claim_lower for w in ['target', 'goal', 'commit', 'plan', 'aim']))
        if has_percent and (not has_baseline):
            esg_type = prediction_label if prediction_label else 'ESG'
            return f'Quantified {esg_type} claim detected (percentage figure present). Risk is elevated because no baseline year is cited — auditor should verify: what is the reference year for this reduction? Words that raised risk: {risk_token_strs}. Words that lowered risk: {safe_token_strs}.'
        if has_commitment and (not has_year):
            return f'Forward commitment claim detected without a specific target year. Risk is elevated — GRI 305 and TCFD require time-bound targets. Words that raised risk: {risk_token_strs}. Words that lowered risk: {safe_token_strs}.'
        return f'This claim was flagged. Words that raised the risk score: {risk_token_strs}. Words that lowered the risk score: {safe_token_strs}. Auditor action: verify supporting data tables in the same document.'