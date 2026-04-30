"""Metrics helpers shared by training, evaluation, and inference flows."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to class probabilities."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape (B, 2), received {tuple(logits.shape)}.")
    return torch.softmax(logits, dim=1)


def positive_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return P(fake) from two-class logits."""
    return probabilities_from_logits(logits)[:, 1]


def _to_numpy(values: Iterable[float] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(list(values), dtype=np.float64)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    index = 0
    while index < sorted_values.shape[0]:
        end = index
        while end + 1 < sorted_values.shape[0] and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        average_rank = (index + end + 2) / 2.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1

    return ranks


def roc_auc_score_binary(
    y_true: Iterable[int] | np.ndarray | torch.Tensor,
    y_score: Iterable[float] | np.ndarray | torch.Tensor,
) -> float | None:
    """Compute binary ROC-AUC from positive-class scores."""
    targets = _to_numpy(y_true).astype(np.int64, copy=False)
    scores = _to_numpy(y_score)

    if targets.shape[0] != scores.shape[0]:
        raise ValueError("Targets and scores must have the same length.")

    positives = targets == 1
    negatives = targets == 0
    num_positives = int(positives.sum())
    num_negatives = int(negatives.sum())

    if num_positives == 0 or num_negatives == 0:
        return None

    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[positives].sum())
    auc = (
        positive_rank_sum - (num_positives * (num_positives + 1) / 2.0)
    ) / (num_positives * num_negatives)
    return float(auc)


def binary_classification_metrics(
    y_true: Iterable[int] | np.ndarray | torch.Tensor,
    y_score: Iterable[float] | np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compute thresholded metrics from positive-class probabilities."""
    targets = _to_numpy(y_true).astype(np.int64, copy=False)
    scores = _to_numpy(y_score)

    if targets.shape[0] != scores.shape[0]:
        raise ValueError("Targets and scores must have the same length.")
    if targets.size == 0:
        raise ValueError("Cannot compute metrics on an empty target set.")

    predictions = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((predictions == 1) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))

    total = targets.shape[0]
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    auc_roc = roc_auc_score_binary(targets, scores)

    return {
        "threshold": float(threshold),
        "num_examples": int(total),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "false_positive_rate": float(false_positive_rate),
    }
