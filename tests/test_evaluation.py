import numpy as np

from urop_healthgraph.evaluation import (
    METRIC_KEYS,
    classification_metrics,
    curve_points,
    select_threshold,
)


def test_metrics_output_schema() -> None:
    truth = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.2, 0.6, 0.4, 0.75, 0.9])
    threshold = select_threshold(truth, probability)
    metrics = classification_metrics(truth, probability, threshold)
    assert set(METRIC_KEYS).issubset(metrics)
    assert len(metrics["confusion_matrix"]) == 2
    curves = curve_points(truth, probability)
    assert set(curves) == {"roc", "precision_recall"}
    assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"])
