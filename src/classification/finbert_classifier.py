"""
src/classification/finbert_classifier.py
==========================================
ESG claim classifier using 

Model: yiyanghkust/finbert-esg (FinBERT fine-tuned specifically for ESG classification)
This model natively outputs Environmental / Social / Governance / None labels.
It was trained on ESG-specific corporate text, so it correctly rejects generic
corporate boilerplate and only fires on actual ESG claims.
Fallback: ProsusAI/finbert (pretrained sentiment, used if ESG model unavailable)

Output labels:
  - esg_label:          "E" | "S" | "G" | "MIXED"
  - risk_score:         0.0 (low risk) – 1.0 (high risk)
  - consistency_flag:   "LIKELY_CONSISTENT" | "NEEDS_REVIEW" | "HIGH_RISK"
  - confidence:         model softmax confidence for top label

Usage:
    clf = FinBertClassifier()
    result = clf.classify("We reduced carbon emissions by 40% since 2019.")
    print(result.esg_label, result.risk_score, result.consistency_flag)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import hf_cfg


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Output of FinBERT classification for one claim."""
    claim_text: str
    esg_label: str                    # "E" | "S" | "G" | "MIXED"
    risk_score: float                 # 0.0–1.0
    consistency_flag: str             # "LIKELY_CONSISTENT" | "NEEDS_REVIEW" | "HIGH_RISK"
    confidence: float                 # model confidence in top label
    raw_scores: dict[str, float] = None   # all label probabilities

    def __post_init__(self):
        if self.raw_scores is None:
            self.raw_scores = {}

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "esg_label": self.esg_label,
            "risk_score": round(self.risk_score, 4),
            "consistency_flag": self.consistency_flag,
            "confidence": round(self.confidence, 4),
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
        }


# ── Label mappings ────────────────────────────────────────────────────────────
# yiyanghkust/finbert-esg outputs these label names natively.
# We normalise them to our internal short codes (E / S / G / MIXED).

ESG_LABEL_MAP = {
    # yiyanghkust/finbert-esg native labels
    "Environmental": "E",
    "Social": "S",
    "Governance": "G",
    "None": "MIXED",
    # Fallback: generic LABEL_N form in case model config doesn't set names
    "LABEL_0": "E",
    "LABEL_1": "S",
    "LABEL_2": "G",
    "LABEL_3": "MIXED",
    # lowercase variants
    "environmental": "E",
    "social": "S",
    "governance": "G",
    "none": "MIXED",
    "mixed": "MIXED",
}

# Risk scores per ESG category:
#   E (climate/env) and G (governance) carry higher audit risk than S.
#   MIXED/None = generic boilerplate → highest uncertainty.
ESG_BASE_RISK = {"E": 0.35, "S": 0.30, "G": 0.55, "MIXED": 0.70}

# Pretrained FinBERT (ProsusAI/finbert) outputs positive/negative/neutral.
# Used ONLY as a fallback when the ESG model is unavailable.
PRETRAINED_SENTIMENT_RISK = {
    "positive": 0.2,
    "negative": 0.7,
    "neutral":  0.45,
}

# Section name → E/S/G label (fallback for pretrained model only)
SECTION_TO_ESG: dict[str, str] = {
    "environmental": "E",
    "social":        "S",
    "governance":    "G",
    "general":       "MIXED",
}


# ── Main class ────────────────────────────────────────────────────────────────

class FinBertClassifier:
    """
    Classifies ESG claims using yiyanghkust/finbert-esg.

    This model was specifically fine-tuned for ESG classification and natively
    outputs Environmental / Social / Governance / None. It correctly rejects
    generic corporate boilerplate that the base ProsusAI/finbert would pass.

    Falls back to ProsusAI/finbert with heuristic section-based labelling if
    the ESG model is unavailable (e.g. no internet access).

    Device: CPU-only (no CUDA required).
    """

    MAX_LENGTH = 512    # FinBERT / BERT max token length

    def __init__(self, use_finetuned: bool = True) -> None:
        """
        Load the classifier.

        Args:
            use_finetuned: Kept for API compatibility. Always attempts to load
                           yiyanghkust/finbert-esg first, then falls back to
                           ProsusAI/finbert if unavailable.
        """
        self.device = torch.device("cpu")
        self.model_id, self.is_finetuned = self._resolve_model_id(use_finetuned)
        logger.info(f"Loading FinBERT from '{self.model_id}' on CPU")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.eval()
            self.model.to(self.device)
            logger.success(f"FinBERT loaded | {self.model_id} | is_finetuned={self.is_finetuned}")
        except Exception as exc:
            logger.error(f"Failed to load model '{self.model_id}': {exc}")
            raise

        # Cache raw label names from model config for display
        self._id2label = getattr(self.model.config, 'id2label', {})

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, claim_text: str, source_section: str = "general") -> ClassificationResult:
        """
        Classify a single ESG claim.

        Args:
            claim_text:     Raw sentence from the ESG filing
            source_section: Section where the claim was found ("environmental"/"social"/"governance"/"general")
                            Used to derive E/S/G label when model is not fine-tuned.

        Returns:
            ClassificationResult with ESG label, risk score, consistency flag
        """
        inputs = self.tokenizer(
            claim_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)[0]
        probs_dict = {
            self.model.config.id2label.get(i, f"LABEL_{i}"): probs[i].item()
            for i in range(len(probs))
        }

        top_label_raw = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[top_label_raw]

        esg_label, risk_score = self._map_to_esg(top_label_raw, probs_dict, source_section)
        consistency_flag = self._compute_consistency_flag(risk_score, confidence)

        return ClassificationResult(
            claim_text=claim_text,
            esg_label=esg_label,
            risk_score=round(risk_score, 4),
            consistency_flag=consistency_flag,
            confidence=round(confidence, 4),
            raw_scores=probs_dict,
        )

    def classify_batch(
        self,
        claims: list[str],
        source_sections: list[str] | None = None,
        batch_size: int = 16,
    ) -> list[ClassificationResult]:
        """
        Classify a list of claims with batching for efficiency.

        Args:
            claims:          List of claim sentences
            source_sections: Parallel list of section names for each claim.
                             Used to assign E/S/G label when not fine-tuned.
            batch_size:      Number of claims to process per forward pass

        Returns:
            List of ClassificationResult (same order as input)
        """
        if source_sections is None:
            source_sections = ["general"] * len(claims)

        results = []
        for i in range(0, len(claims), batch_size):
            batch = claims[i : i + batch_size]
            batch_sections = source_sections[i : i + batch_size]
            logger.debug(f"Classifying batch {i//batch_size + 1} ({len(batch)} claims)")

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_LENGTH,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs_batch = F.softmax(outputs.logits, dim=-1)

            for j, (claim_text, section) in enumerate(zip(batch, batch_sections)):
                probs = probs_batch[j]
                probs_dict = {
                    self.model.config.id2label.get(k, f"LABEL_{k}"): probs[k].item()
                    for k in range(len(probs))
                }
                top_label_raw = max(probs_dict, key=probs_dict.get)
                confidence = probs_dict[top_label_raw]
                esg_label, risk_score = self._map_to_esg(top_label_raw, probs_dict, section)
                consistency_flag = self._compute_consistency_flag(risk_score, confidence)

                results.append(ClassificationResult(
                    claim_text=claim_text,
                    esg_label=esg_label,
                    risk_score=round(risk_score, 4),
                    consistency_flag=consistency_flag,
                    confidence=round(confidence, 4),
                    raw_scores=probs_dict,
                ))

        logger.info(f"Batch classification complete: {len(results)} claims")
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_model_id(self, use_finetuned: bool) -> tuple[str, bool]:
        """
        Return (model_id, is_finetuned).

        Priority:
        1. yiyanghkust/finbert-esg  (the public ESG fine-tuned model — our default)
        2. ProsusAI/finbert         (pretrained fallback with heuristic labelling)
        """
        primary = "yiyanghkust/finbert-esg"
        fallback = "ProsusAI/finbert"

        if not use_finetuned:
            return fallback, False

        try:
            from huggingface_hub import model_info
            model_info(primary)
            logger.info(f"ESG model available: {primary}")
            return primary, True
        except Exception:
            logger.warning(
                f"'{primary}' not reachable. Falling back to '{fallback}'. "
                "Check your internet connection or HF_TOKEN."
            )
            return fallback, False

    def _build_label_map(self) -> dict:
        """Kept for API compatibility — label resolution now done in _map_to_esg."""
        return {}

    def _map_to_esg(
        self,
        raw_label: str,
        probs_dict: dict[str, float],
        source_section: str = "general",
    ) -> tuple[str, float]:
        """
        Map a raw model label + probabilities to (esg_label, risk_score).

        For yiyanghkust/finbert-esg (is_finetuned=True):
          - Directly maps Environmental/Social/Governance/None → E/S/G/MIXED
          - Uses ESG_BASE_RISK + confidence penalty for risk score
          - "None" label (MIXED) means the sentence is NOT an ESG claim →
            high risk score to flag it for review/filtering

        For pretrained ProsusAI/finbert (fallback):
          - Uses source_section to derive E/S/G label
          - Uses sentiment probabilities for risk score estimate
        """
        if self.is_finetuned:
            esg_label = ESG_LABEL_MAP.get(raw_label, "MIXED")
            base_risk = ESG_BASE_RISK.get(esg_label, 0.5)
            top_confidence = max(probs_dict.values())
            # Penalise low-confidence predictions (uncertain = higher risk)
            risk_score = base_risk + (1.0 - top_confidence) * 0.2
            return esg_label, min(1.0, risk_score)
        else:
            # Pretrained FinBERT: sentiment label drives risk, section drives E/S/G
            esg_label = SECTION_TO_ESG.get(source_section.lower(), "MIXED")
            risk_score = PRETRAINED_SENTIMENT_RISK.get(raw_label, 0.45)
            top_confidence = max(probs_dict.values())
            risk_score = min(1.0, risk_score + (1.0 - top_confidence) * 0.15)
            return esg_label, risk_score

    @staticmethod
    def _compute_consistency_flag(risk_score: float, confidence: float) -> str:
        """
        Derive a human-readable consistency flag from risk score + confidence.
        """
        if risk_score >= 0.7 or (risk_score >= 0.5 and confidence < 0.6):
            return "HIGH_RISK"
        elif risk_score >= 0.4 or confidence < 0.7:
            return "NEEDS_REVIEW"
        else:
            return "LIKELY_CONSISTENT"
