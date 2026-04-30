"""Unit tests for shared binary classification metrics."""

import torch

from src.utils.metrics import (
    binary_classification_metrics,
    positive_class_probabilities,
    roc_auc_score_binary,
)


def test_positive_class_probabilities_extracts_fake_scores() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    fake_scores = positive_class_probabilities(logits)

    assert fake_scores.shape == (2,)
    assert fake_scores[0] < 0.5
    assert fake_scores[1] > 0.5


def test_binary_classification_metrics_match_perfect_predictions() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]

    metrics = binary_classification_metrics(labels, scores, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["auc_roc"] == 1.0


def test_roc_auc_score_binary_returns_none_for_single_class_targets() -> None:
    auc = roc_auc_score_binary([1, 1, 1], [0.8, 0.9, 0.7])
    assert auc is None
