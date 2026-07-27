import numpy as np
import pytest

from urop_healthgraph.statistical_analysis import (
    BOOTSTRAP_METRICS,
    generate_cluster_resample_indices,
    patient_clustered_paired_bootstrap,
)


def _models(probabilities: np.ndarray) -> dict:
    return {
        slug: {
            "name": slug,
            "probabilities": probabilities.copy(),
            "threshold": 0.5,
        }
        for slug in ("logistic_regression", "pca_knn", "gcn")
    }


def test_cluster_resampling_preserves_complete_patient_groups() -> None:
    patients = np.asarray(["A", "A", "B", "C", "C", "C"])
    for indices in generate_cluster_resample_indices(patients, 30, 17):
        counts = np.bincount(indices, minlength=len(patients))
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]


def test_paired_bootstrap_schema_resamples_models_identically() -> None:
    truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    patients = np.asarray(["A", "A", "B", "C", "D", "E", "F", "F"])
    probabilities = np.asarray([0.1, 0.7, 0.2, 0.8, 0.4, 0.6, 0.3, 0.9])
    confidence, paired = patient_clustered_paired_bootstrap(
        truth, patients, _models(probabilities), replicates=80, random_seed=11
    )
    assert set(confidence) == {"metadata", "models"}
    assert set(confidence["models"]) == {
        "logistic_regression",
        "pca_knn",
        "gcn",
    }
    assert confidence["metadata"]["paired_resampling"] is True
    assert len(confidence["metadata"]["resample_index_sha256"]) == 64
    for comparison in paired["comparisons"]:
        for metric in BOOTSTRAP_METRICS:
            assert comparison["metrics"][metric]["observed_difference"] == 0.0
            assert comparison["metrics"][metric]["bootstrap_mean_difference"] == 0.0


def test_invalid_single_class_replicates_are_skipped_and_bounds_ordered() -> None:
    truth = np.asarray([1, 0, 0])
    patients = np.asarray(["positive", "negative-a", "negative-b"])
    probabilities = np.asarray([0.8, 0.2, 0.3])
    confidence, _ = patient_clustered_paired_bootstrap(
        truth, patients, _models(probabilities), replicates=200, random_seed=3
    )
    metadata = confidence["metadata"]
    assert metadata["invalid_single_class_replicates"] > 0
    assert (
        metadata["valid_replicates"]
        + metadata["invalid_single_class_replicates"]
        == 200
    )
    for model in confidence["models"].values():
        for interval in model["intervals"].values():
            assert interval["ci_95_lower"] <= interval["ci_95_upper"]


def test_all_invalid_single_class_bootstrap_fails_explicitly() -> None:
    truth = np.zeros(4, dtype=int)
    patients = np.asarray(["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="No valid two-class"):
        patient_clustered_paired_bootstrap(
            truth,
            patients,
            _models(np.asarray([0.1, 0.2, 0.3, 0.4])),
            replicates=10,
        )
