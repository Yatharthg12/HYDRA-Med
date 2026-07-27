"""Patient-clustered uncertainty and repeated-experiment summaries."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .evaluation import classification_metrics


BOOTSTRAP_METRICS = (
    "pr_auc",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)

DEFAULT_MODEL_PAIRS = (
    ("logistic_regression", "gcn"),
    ("logistic_regression", "pca_knn"),
    ("gcn", "pca_knn"),
)

GCN_STABILITY_METRICS = (
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def patient_cluster_groups(patient_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return sorted patients and all encounter positions belonging to each one."""
    patients = np.asarray(patient_ids).astype(str)
    unique = np.unique(patients)
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    return unique, groups


def generate_cluster_resample_indices(
    patient_ids: np.ndarray, replicates: int, random_seed: int
) -> list[np.ndarray]:
    """Generate encounter indices by resampling complete patient clusters.

    If a patient is sampled more than once, every encounter in that patient's
    cluster is repeated the same number of times in the returned index vector.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    _, groups = patient_cluster_groups(patient_ids)
    if not groups:
        raise ValueError("patient_ids must not be empty")
    rng = np.random.default_rng(random_seed)
    resamples = []
    for _ in range(replicates):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        resamples.append(np.concatenate([groups[index] for index in sampled_groups]))
    return resamples


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    """Return an ordered 95% percentile interval."""
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("Cannot calculate an interval from no values")
    lower, upper = np.percentile(array, [2.5, 97.5])
    return float(min(lower, upper)), float(max(lower, upper))


def descriptive_summary(values: Iterable[float]) -> dict[str, float | int]:
    """Return repeated-run statistics and a two-sided 95% t interval."""
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("Cannot summarize no values")
    count = int(len(array))
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if count > 1 else 0.0
    if count > 1:
        margin = float(
            student_t.ppf(0.975, df=count - 1)
            * standard_deviation
            / np.sqrt(count)
        )
    else:
        margin = 0.0
    return {
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "ci_95_lower": mean - margin,
        "ci_95_upper": mean + margin,
    }


def patient_clustered_paired_bootstrap(
    y_true: np.ndarray,
    patient_ids: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int = 1_000,
    random_seed: int = 42,
    model_pairs: tuple[tuple[str, str], ...] = DEFAULT_MODEL_PAIRS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calculate model CIs and paired differences from common patient resamples.

    ``models`` maps a model slug to ``probabilities``, ``threshold``, and an
    optional display ``name``. Every model is evaluated on the exact same
    encounter indices for a replicate.
    """
    truth = np.asarray(y_true, dtype=int)
    patients = np.asarray(patient_ids).astype(str)
    if len(truth) != len(patients):
        raise ValueError("y_true and patient_ids must have identical lengths")
    if set(models) != {slug for pair in model_pairs for slug in pair}:
        required = {slug for pair in model_pairs for slug in pair}
        missing = required - set(models)
        if missing:
            raise ValueError(f"Missing models required by model_pairs: {sorted(missing)}")
    prepared: dict[str, dict[str, Any]] = {}
    for slug, values in models.items():
        probability = np.asarray(values["probabilities"], dtype=float)
        if len(probability) != len(truth):
            raise ValueError(f"Probability length mismatch for {slug}")
        prepared[slug] = {
            "probabilities": probability,
            "threshold": float(values["threshold"]),
            "name": str(values.get("name", slug)),
        }

    resamples = generate_cluster_resample_indices(patients, replicates, random_seed)
    sample_hash = hashlib.sha256()
    distributions = {
        slug: {metric: [] for metric in BOOTSTRAP_METRICS} for slug in prepared
    }
    paired_distributions = {
        f"{left}_minus_{right}": {metric: [] for metric in BOOTSTRAP_METRICS}
        for left, right in model_pairs
    }
    valid = 0
    invalid = 0
    for indices in resamples:
        sample_hash.update(np.asarray(indices, dtype=np.int64).tobytes())
        sampled_truth = truth[indices]
        if len(np.unique(sampled_truth)) < 2:
            invalid += 1
            continue
        replicate_metrics: dict[str, dict[str, Any]] = {}
        for slug, values in prepared.items():
            metrics = classification_metrics(
                sampled_truth,
                values["probabilities"][indices],
                values["threshold"],
            )
            replicate_metrics[slug] = metrics
            for metric in BOOTSTRAP_METRICS:
                distributions[slug][metric].append(float(metrics[metric]))
        for left, right in model_pairs:
            key = f"{left}_minus_{right}"
            for metric in BOOTSTRAP_METRICS:
                paired_distributions[key][metric].append(
                    float(replicate_metrics[left][metric])
                    - float(replicate_metrics[right][metric])
                )
        valid += 1
    if valid == 0:
        raise ValueError("No valid two-class patient bootstrap replicates were produced")

    confidence_models = {}
    observed_metrics = {}
    for slug, values in prepared.items():
        observed = classification_metrics(
            truth, values["probabilities"], values["threshold"]
        )
        observed_metrics[slug] = observed
        intervals = {}
        for metric in BOOTSTRAP_METRICS:
            lower, upper = percentile_interval(distributions[slug][metric])
            intervals[metric] = {
                "estimate": float(observed[metric]),
                "bootstrap_mean": float(np.mean(distributions[slug][metric])),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
            }
        confidence_models[slug] = {
            "name": values["name"],
            "threshold": values["threshold"],
            "intervals": intervals,
        }

    comparisons = []
    for left, right in model_pairs:
        key = f"{left}_minus_{right}"
        metrics = {}
        for metric in BOOTSTRAP_METRICS:
            differences = np.asarray(paired_distributions[key][metric], dtype=float)
            lower, upper = percentile_interval(differences)
            probability_above = float(np.mean(differences > 0))
            non_positive = (float(np.sum(differences <= 0)) + 1.0) / (
                len(differences) + 1.0
            )
            non_negative = (float(np.sum(differences >= 0)) + 1.0) / (
                len(differences) + 1.0
            )
            metrics[metric] = {
                "observed_difference": float(
                    observed_metrics[left][metric] - observed_metrics[right][metric]
                ),
                "bootstrap_mean_difference": float(np.mean(differences)),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "proportion_above_zero": probability_above,
                "two_sided_bootstrap_tail_probability": float(
                    min(1.0, 2.0 * min(non_positive, non_negative))
                ),
                "interval_excludes_zero": bool(lower > 0 or upper < 0),
            }
        comparisons.append(
            {
                "comparison": key,
                "left_model": left,
                "left_name": prepared[left]["name"],
                "right_model": right,
                "right_name": prepared[right]["name"],
                "metrics": metrics,
            }
        )

    metadata = {
        "method": "patient-clustered paired percentile bootstrap",
        "master_seed": int(random_seed),
        "requested_replicates": int(replicates),
        "valid_replicates": int(valid),
        "invalid_single_class_replicates": int(invalid),
        "unique_test_patients": int(len(np.unique(patients))),
        "test_encounters": int(len(truth)),
        "paired_resampling": True,
        "resample_index_sha256": sample_hash.hexdigest(),
        "confidence_level": 0.95,
    }
    confidence = {"metadata": metadata, "models": confidence_models}
    paired = {
        "metadata": {
            **metadata,
            "probability_note": (
                "The two-sided bootstrap tail probability is a descriptive "
                "paired-resampling measure, not proof of clinical significance."
            ),
        },
        "comparisons": comparisons,
    }
    return confidence, paired


def summarize_gcn_seed_results(
    seed_results: list[dict[str, Any]], primary_seed: int
) -> dict[str, Any]:
    """Summarize prespecified GCN seed runs without selecting a best seed."""
    if not seed_results:
        raise ValueError("At least one GCN seed result is required")
    seeds = [int(row["seed"]) for row in seed_results]
    if primary_seed not in seeds:
        raise ValueError("The primary prespecified seed is missing")
    return {
        "metadata": {
            "primary_seed": int(primary_seed),
            "seeds": seeds,
            "number_of_seeds": len(seeds),
            "selection_policy": (
                "The primary comparison remains the prespecified primary seed. "
                "No best-seed selection is performed."
            ),
            "interval_method": (
                "Two-sided 95% Student-t interval across deterministic training seeds."
            ),
        },
        "runs": seed_results,
        "summary": {
            metric: descriptive_summary(
                row["test_metrics"][metric] for row in seed_results
            )
            for metric in GCN_STABILITY_METRICS
        },
    }
