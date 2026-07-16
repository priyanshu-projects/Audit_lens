"""
tests/test_finbert_classifier.py
==================================
Tests for FinBERT classifier — uses a mock model to avoid loading weights in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestFinBertClassifier:

    @pytest.fixture
    def mock_classifier(self):
        """Create a FinBertClassifier with a fully mocked model."""
        with patch("src.classification.finbert_classifier.AutoModelForSequenceClassification.from_pretrained") as mock_model_cls, \
             patch("src.classification.finbert_classifier.AutoTokenizer.from_pretrained") as mock_tok_cls, \
             patch("src.classification.finbert_classifier.FinBertClassifier._resolve_model_id") as mock_resolve:

            mock_resolve.return_value = ("ProsusAI/finbert", False)

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
            mock_tok_cls.return_value = mock_tokenizer

            # Mock model
            mock_model = MagicMock()
            mock_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
            mock_logits = torch.tensor([[2.0, 0.5, 0.3]])  # positive wins
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            mock_model_cls.return_value = mock_model

            from src.classification.finbert_classifier import FinBertClassifier
            clf = FinBertClassifier(use_finetuned=False)

        return clf

    def test_classify_returns_result(self, mock_classifier):
        from src.classification.finbert_classifier import ClassificationResult
        result = mock_classifier.classify("We reduced emissions by 40% since 2019.")
        assert isinstance(result, ClassificationResult)

    def test_classify_result_has_required_fields(self, mock_classifier):
        result = mock_classifier.classify("We reduced emissions by 40%.")
        assert isinstance(result.esg_label, str)
        assert isinstance(result.risk_score, float)
        assert isinstance(result.consistency_flag, str)
        assert isinstance(result.confidence, float)

    def test_risk_score_in_range(self, mock_classifier):
        result = mock_classifier.classify("We committed to net-zero by 2040.")
        assert 0.0 <= result.risk_score <= 1.0

    def test_confidence_in_range(self, mock_classifier):
        result = mock_classifier.classify("We reduced emissions by 40%.")
        assert 0.0 <= result.confidence <= 1.0

    def test_consistency_flag_valid_values(self, mock_classifier):
        result = mock_classifier.classify("We reduced emissions by 40%.")
        assert result.consistency_flag in ("LIKELY_CONSISTENT", "NEEDS_REVIEW", "HIGH_RISK")

    def test_esg_label_valid_values(self, mock_classifier):
        result = mock_classifier.classify("We reduced Scope 1 emissions.")
        assert result.esg_label in ("E", "S", "G", "MIXED")

    def test_to_dict_returns_dict(self, mock_classifier):
        result = mock_classifier.classify("We reduced emissions by 40%.")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "esg_label" in d
        assert "risk_score" in d
        assert "consistency_flag" in d

    def test_consistency_flag_logic(self):
        from src.classification.finbert_classifier import FinBertClassifier
        # Test the static method directly
        assert FinBertClassifier._compute_consistency_flag(0.8, 0.9) == "HIGH_RISK"
        assert FinBertClassifier._compute_consistency_flag(0.5, 0.5) == "HIGH_RISK"
        assert FinBertClassifier._compute_consistency_flag(0.4, 0.75) == "NEEDS_REVIEW"
        assert FinBertClassifier._compute_consistency_flag(0.1, 0.9) == "LIKELY_CONSISTENT"
