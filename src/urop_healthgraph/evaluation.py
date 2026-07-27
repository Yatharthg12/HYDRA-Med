"""Shared threshold selection, predictive metrics, and curve generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the validation threshold that maximizes positive-class F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(scores))])


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Calculate the common imbalanced-classification metric schema."""
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    predicted = (probability >= threshold).astype(int)
    roc_auc = float(roc_auc_score(truth, probability)) if len(np.unique(truth)) > 1 else None
    pr_auc = (
        float(average_precision_score(truth, probability))
        if len(np.unique(truth)) > 1
        else None
    )
    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "support": int(len(truth)),
        "positive_support": int(truth.sum()),
    }


def curve_points(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Return downsampled ROC and precision-recall coordinates."""
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    fpr, tpr, roc_thresholds = roc_curve(truth, probability)
    precision, recall, pr_thresholds = precision_recall_curve(truth, probability)

    def sampled(*arrays: np.ndarray, maximum: int = 250) -> list[np.ndarray]:
        length = len(arrays[0])
        indices = np.unique(np.linspace(0, length - 1, min(maximum, length), dtype=int))
        return [array[indices] for array in arrays]

    fpr_s, tpr_s, roc_t_s = sampled(fpr, tpr, roc_thresholds)
    precision_s, recall_s = sampled(precision, recall)
    return {
        "roc": {
            "fpr": fpr_s.tolist(),
            "tpr": tpr_s.tolist(),
            "thresholds": roc_t_s.tolist(),
        },
        "precision_recall": {
            "recall": recall_s.tolist(),
            "precision": precision_s.tolist(),
            "thresholds": pr_thresholds.tolist()[:250],
        },
    }
