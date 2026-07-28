from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.settings import hf_cfg

@dataclass
class ClassificationResult:
    claim_text: str
    esg_label: str
    risk_score: float
    consistency_flag: str
    confidence: float
    raw_scores: dict[str, float] = None

    def __post_init__(self):
        if self.raw_scores is None:
            self.raw_scores = {}

    def to_dict(self) -> dict:
        return {'claim_text': self.claim_text, 'esg_label': self.esg_label, 'risk_score': round(self.risk_score, 4), 'consistency_flag': self.consistency_flag, 'confidence': round(self.confidence, 4), 'raw_scores': {k: round(v, 4) for k, v in self.raw_scores.items()}}
ESG_LABEL_MAP = {'Environmental': 'E', 'Social': 'S', 'Governance': 'G', 'None': 'MIXED', 'LABEL_0': 'E', 'LABEL_1': 'S', 'LABEL_2': 'G', 'LABEL_3': 'MIXED', 'environmental': 'E', 'social': 'S', 'governance': 'G', 'none': 'MIXED', 'mixed': 'MIXED'}
ESG_BASE_RISK = {'E': 0.35, 'S': 0.3, 'G': 0.55, 'MIXED': 0.7}
PRETRAINED_SENTIMENT_RISK = {'positive': 0.2, 'negative': 0.7, 'neutral': 0.45}
SECTION_TO_ESG: dict[str, str] = {'environmental': 'E', 'social': 'S', 'governance': 'G', 'general': 'MIXED'}

class FinBertClassifier:
    MAX_LENGTH = 512

    def __init__(self, use_finetuned: bool=True) -> None:
        self.device = torch.device('cpu')
        self.model_id, self.is_finetuned = self._resolve_model_id(use_finetuned)
        logger.info(f"Loading FinBERT from '{self.model_id}' on CPU")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.eval()
            self.model.to(self.device)
            logger.success(f'FinBERT loaded | {self.model_id} | is_finetuned={self.is_finetuned}')
        except Exception as exc:
            logger.error(f"Failed to load model '{self.model_id}': {exc}")
            raise
        self._id2label = getattr(self.model.config, 'id2label', {})

    def classify(self, claim_text: str, source_section: str='general') -> ClassificationResult:
        inputs = self.tokenizer(claim_text, return_tensors='pt', truncation=True, max_length=self.MAX_LENGTH, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        probs_dict = {self.model.config.id2label.get(i, f'LABEL_{i}'): probs[i].item() for i in range(len(probs))}
        top_label_raw = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[top_label_raw]
        esg_label, risk_score = self._map_to_esg(top_label_raw, probs_dict, source_section)
        consistency_flag = self._compute_consistency_flag(risk_score, confidence)
        return ClassificationResult(claim_text=claim_text, esg_label=esg_label, risk_score=round(risk_score, 4), consistency_flag=consistency_flag, confidence=round(confidence, 4), raw_scores=probs_dict)

    def classify_batch(self, claims: list[str], source_sections: list[str] | None=None, batch_size: int=16) -> list[ClassificationResult]:
        if source_sections is None:
            source_sections = ['general'] * len(claims)
        results = []
        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            batch_sections = source_sections[i:i + batch_size]
            logger.debug(f'Classifying batch {i // batch_size + 1} ({len(batch)} claims)')
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=self.MAX_LENGTH, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs_batch = F.softmax(outputs.logits, dim=-1)
            for j, (claim_text, section) in enumerate(zip(batch, batch_sections)):
                probs = probs_batch[j]
                probs_dict = {self.model.config.id2label.get(k, f'LABEL_{k}'): probs[k].item() for k in range(len(probs))}
                top_label_raw = max(probs_dict, key=probs_dict.get)
                confidence = probs_dict[top_label_raw]
                esg_label, risk_score = self._map_to_esg(top_label_raw, probs_dict, section)
                consistency_flag = self._compute_consistency_flag(risk_score, confidence)
                results.append(ClassificationResult(claim_text=claim_text, esg_label=esg_label, risk_score=round(risk_score, 4), consistency_flag=consistency_flag, confidence=round(confidence, 4), raw_scores=probs_dict))
        logger.info(f'Batch classification complete: {len(results)} claims')
        return results

    def _resolve_model_id(self, use_finetuned: bool) -> tuple[str, bool]:
        primary = 'yiyanghkust/finbert-esg'
        fallback = 'ProsusAI/finbert'
        if not use_finetuned:
            return (fallback, False)
        try:
            from huggingface_hub import model_info
            model_info(primary)
            logger.info(f'ESG model available: {primary}')
            return (primary, True)
        except Exception:
            logger.warning(f"'{primary}' not reachable. Falling back to '{fallback}'. Check your internet connection or HF_TOKEN.")
            return (fallback, False)

    def _build_label_map(self) -> dict:
        return {}

    def _map_to_esg(self, raw_label: str, probs_dict: dict[str, float], source_section: str='general') -> tuple[str, float]:
        if self.is_finetuned:
            esg_label = ESG_LABEL_MAP.get(raw_label, 'MIXED')
            base_risk = ESG_BASE_RISK.get(esg_label, 0.5)
            top_confidence = max(probs_dict.values())
            risk_score = base_risk + (1.0 - top_confidence) * 0.2
            return (esg_label, min(1.0, risk_score))
        else:
            esg_label = SECTION_TO_ESG.get(source_section.lower(), 'MIXED')
            risk_score = PRETRAINED_SENTIMENT_RISK.get(raw_label, 0.45)
            top_confidence = max(probs_dict.values())
            risk_score = min(1.0, risk_score + (1.0 - top_confidence) * 0.15)
            return (esg_label, risk_score)

    @staticmethod
    def _compute_consistency_flag(risk_score: float, confidence: float) -> str:
        if risk_score >= 0.7 or (risk_score >= 0.5 and confidence < 0.6):
            return 'HIGH_RISK'
        elif risk_score >= 0.4 or confidence < 0.7:
            return 'NEEDS_REVIEW'
        else:
            return 'LIKELY_CONSISTENT'