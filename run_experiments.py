"""Run the complete reproducible HealthGraph research experiment."""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urop_healthgraph.artifacts import read_json, write_json
from urop_healthgraph.baseline_models import (
    train_logistic_regression,
    train_pca_knn,
)
from urop_healthgraph.config import ExperimentConfig
from urop_healthgraph.data_processing import (
    build_data_dictionary,
    clean_dataset,
    dataset_variant_summary,
    distribution_summary,
    load_dataset,
    parse_ids_mapping,
)
from urop_healthgraph.evaluation import METRIC_KEYS
from urop_healthgraph.feature_engineering import (
    MEDICATION_COLUMNS,
    build_feature_frame,
    make_preprocessor,
)
from urop_healthgraph.gcn_model import train_gcn
from urop_healthgraph.graph_ablation import run_graph_ablation_study
from urop_healthgraph.graph_builder import (
    build_heterogeneous_graph_summary,
    build_similarity_graph,
    fit_relation_schema,
    neighbor_evidence,
    normalized_adjacency,
    relation_token_sets,
)
from urop_healthgraph.robustness import (
    ROBUSTNESS_METRICS,
    run_repeated_robustness,
)
from urop_healthgraph.split_manager import (
    create_patient_splits,
    save_split_assignments,
    split_summary,
)
from urop_healthgraph.statistical_analysis import (
    BOOTSTRAP_METRICS,
    GCN_STABILITY_METRICS,
    patient_clustered_paired_bootstrap,
    summarize_gcn_seed_results,
)
from urop_healthgraph.warshall import warshall_iterations


LOGGER = logging.getLogger("healthgraph.experiments")
TOTAL_STAGES = 15


def stage(number: int, label: str, function: Callable[[], Any]) -> Any:
    """Run and time one visible pipeline stage."""
    LOGGER.info("[%d/%d] %s", number, TOTAL_STAGES, label)
    started = time.perf_counter()
    result = function()
    LOGGER.info("      completed in %.2f s", time.perf_counter() - started)
    return result


def _active_medications(row: pd.Series) -> str:
    return "; ".join(
        medication
        for medication in MEDICATION_COLUMNS
        if int(row.get(f"med_{medication}_active", 0) or 0)
    )


def _save_figures(
    comparison: list[dict[str, Any]],
    robustness_summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["name"] for row in comparison]
    x = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        x - width / 2,
        [row["test_metrics"]["pr_auc"] for row in comparison],
        width,
        label="PR-AUC",
        color="#35d0ba",
    )
    axis.bar(
        x + width / 2,
        [row["test_metrics"]["roc_auc"] for row in comparison],
        width,
        label="ROC-AUC",
        color="#7c8cff",
    )
    axis.set_xticks(x, labels, rotation=10, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Patient-safe test model discrimination")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    scenario_labels = [row["label"] for row in robustness_summary]
    for metric, color in (("pr_auc", "#35d0ba"), ("roc_auc", "#7c8cff")):
        means = [row["metrics"][metric]["mean"] for row in robustness_summary]
        lower = [
            mean - row["metrics"][metric]["ci_95_lower"]
            for mean, row in zip(means, robustness_summary, strict=True)
        ]
        upper = [
            row["metrics"][metric]["ci_95_upper"] - mean
            for mean, row in zip(means, robustness_summary, strict=True)
        ]
        axis.errorbar(
            scenario_labels,
            means,
            yerr=np.asarray([lower, upper]),
            marker="o",
            capsize=4,
            label=metric.upper().replace("_", "-"),
            color=color,
        )
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=18)
    axis.set_ylabel("Mean score with 95% t interval")
    axis.set_title("Repeated GCN relationship robustness")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "robustness.png", dpi=150)
    plt.close(fig)


def _comparison_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        models,
        key=lambda model: (
            model["test_metrics"]["pr_auc"] or -1,
            model["test_metrics"]["roc_auc"] or -1,
        ),
        reverse=True,
    )
    for rank, model in enumerate(ranked, start=1):
        model["rank_by_test_pr_auc"] = rank
    return models


def _gcn_seed_record(seed: int, result: dict[str, Any], primary: bool) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "is_primary_prespecified_seed": bool(primary),
        "selected_epoch": int(result["best_epoch"]),
        "epochs_completed": int(result["epochs_completed"]),
        "selected_threshold": float(result["validation_selected_threshold"]),
        "training_seconds": float(result["training_seconds"]),
        "inference_seconds": float(result["inference_seconds"]),
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "confusion_matrix": result["test_metrics"]["confusion_matrix"],
        "training_history": result["history"],
    }


def _bootstrap_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for slug, model in payload.get("models", {}).items():
        for metric, values in model["intervals"].items():
            rows.append(
                {
                    "model": slug,
                    "model_name": model["name"],
                    "metric": metric,
                    **values,
                    "valid_replicates": payload["metadata"]["valid_replicates"],
                    "invalid_replicates": payload["metadata"][
                        "invalid_single_class_replicates"
                    ],
                }
            )
    return rows


def _paired_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in payload.get("comparisons", []):
        for metric, values in comparison["metrics"].items():
            rows.append(
                {
                    "comparison": comparison["comparison"],
                    "left_model": comparison["left_model"],
                    "right_model": comparison["right_model"],
                    "metric": metric,
                    **values,
                }
            )
    return rows


def _robustness_summary_csv_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in summaries:
        for metric, values in scenario["metrics"].items():
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "label": scenario["label"],
                    "perturbation_type": scenario["perturbation_type"],
                    "perturbation_level": scenario["perturbation_level"],
                    "number_of_trials": scenario["number_of_trials"],
                    "metric": metric,
                    **values,
                    "edge_count_before": scenario["edge_count_before"],
                    "mean_edge_count_after": scenario["edge_count_after"]["mean"],
                }
            )
    return rows


def _ablation_csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "scenario": result["scenario"],
                "label": result["label"],
                "training_strategy": result["training_strategy"],
                "threshold": result["threshold"],
                "selected_epoch": result["selected_epoch"],
                "training_seconds": result["training_seconds"],
                "test_edge_count": result["test_edge_count"],
                **{
                    metric: result["test_metrics"][metric]
                    for metric in METRIC_KEYS
                },
                "confusion_matrix": result["test_metrics"]["confusion_matrix"],
            }
        )
    return rows


def _legacy_robustness_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Maintain the original aggregate artifact with repeated-trial means."""
    return [
        {
            "scenario": row["scenario"],
            "label": row["label"],
            "perturbation_level": row["perturbation_level"],
            "edge_count_before": row["edge_count_before"],
            "edge_count_after": int(round(row["edge_count_after"]["mean"])),
            "number_of_trials": row["number_of_trials"],
            **{
                metric: row["metrics"][metric]["mean"]
                for metric in ROBUSTNESS_METRICS
            },
        }
        for row in summaries
    ]


def _package_versions() -> dict[str, str]:
    result = {}
    for package in (
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "networkx",
        "Flask",
        "joblib",
        "matplotlib",
        "torch",
        "pytest",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_reproducibility_manifest(
    config: ExperimentConfig,
    profile: str,
    total_seconds: float,
    dataset_summary: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    artifact_hashes = {}
    for path in sorted(config.artifacts_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            artifact_hashes[str(path.relative_to(config.project_root)).replace("\\", "/")] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "execution_timestamp": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "package_versions": _package_versions(),
        "dataset_mode": config.dataset,
        "execution_profile": profile,
        "random_seeds": {
            "primary": config.random_seed,
            "bootstrap_master": config.bootstrap_seed,
            "gcn_stability": list(config.gcn_stability_seeds),
            "robustness_trials": list(config.robustness_trial_seeds),
            "graph_ablation": config.graph_ablation_seed,
        },
        "bootstrap_replicates": int(bootstrap_replicates),
        "eligible_encounters": dataset_summary["eligible_rows"],
        "unique_patients": dataset_summary["unique_patients"],
        "split_counts": dataset_summary["splits"],
        "total_experiment_duration_seconds": float(total_seconds),
        "artifact_hashes": artifact_hashes,
    }
    write_json(config.project_root / "docs" / "reproducibility_manifest.json", manifest)
    return manifest


def _analysis_signature(
    config: ExperimentConfig, profile: str, bootstrap_replicates: int
) -> dict[str, Any]:
    return {
        "profile": profile,
        "bootstrap_replicates": int(bootstrap_replicates),
        "gcn_seeds": (
            list(config.gcn_stability_seeds)
            if profile == "research"
            else [config.random_seed]
        ),
        "robustness_seeds": (
            list(config.robustness_trial_seeds)
            if profile == "research"
            else [config.robustness_trial_seeds[0]]
        ),
        "pca_grid": (
            list(config.pca_component_grid) if profile == "research" else [50]
        ),
        "graph_ablation": profile == "research",
    }


def run(
    config: ExperimentConfig,
    *,
    force: bool = False,
    profile: str = "research",
    bootstrap_replicates: int | None = None,
    skip_bootstrap: bool = False,
) -> dict[str, Any]:
    """Execute core models and the selected statistical research profile."""
    if profile not in {"quick", "research"}:
        raise ValueError("profile must be 'quick' or 'research'")
    config.ensure_directories()
    requested_bootstrap = (
        config.bootstrap_replicates
        if bootstrap_replicates is None
        else int(bootstrap_replicates)
    )
    if requested_bootstrap < 0:
        raise ValueError("bootstrap replicates cannot be negative")
    if skip_bootstrap or (profile == "quick" and bootstrap_replicates is None):
        requested_bootstrap = 0
    signature = _analysis_signature(config, profile, requested_bootstrap)

    manifest_path = config.artifacts_dir / "metrics" / "run_manifest.json"
    comparison_path = config.artifacts_dir / "metrics" / "model_comparison.json"
    manifest = read_json(manifest_path, {})
    if (
        not force
        and comparison_path.exists()
        and manifest.get("dataset_mode") == config.dataset
        and manifest.get("status") == "complete"
        and manifest.get("analysis_signature") == signature
    ):
        LOGGER.info(
            "Verified %s artifacts already exist for %s mode. Use --force to retrain.",
            profile,
            config.dataset,
        )
        return read_json(comparison_path)

    pipeline_started = time.perf_counter()
    raw = stage(1, f"Auditing {config.dataset} dataset", lambda: load_dataset(config))
    mappings = parse_ids_mapping(config.mapping_path)
    cleaned, dataset_summary = clean_dataset(raw, mappings)
    dataset_summary["dataset_mode"] = config.dataset
    dataset_summary["variants"] = dataset_variant_summary(config)
    dataset_summary["distributions"] = distribution_summary(cleaned)
    dataset_summary["data_dictionary"] = build_data_dictionary(
        raw.columns.tolist(), mappings
    )

    assignments = stage(
        2,
        "Creating patient-grouped splits",
        lambda: create_patient_splits(cleaned, config),
    )
    save_split_assignments(cleaned, assignments, config.split_path)
    dataset_summary["splits"] = split_summary(cleaned, assignments)

    features = stage(
        3,
        "Engineering leakage-aware predictors",
        lambda: build_feature_frame(cleaned),
    )
    split_indices = {
        name: np.flatnonzero(assignments.to_numpy() == name)
        for name in ("train", "validation", "test")
    }
    X = {
        name: features.iloc[index].reset_index(drop=True)
        for name, index in split_indices.items()
    }
    y = {
        name: cleaned.iloc[index]["target"].to_numpy(dtype=np.int64)
        for name, index in split_indices.items()
    }
    frames = {
        name: cleaned.iloc[index].reset_index(drop=True)
        for name, index in split_indices.items()
    }

    warshall = stage(4, "Verifying Warshall demonstration", warshall_iterations)
    write_json(config.artifacts_dir / "graphs" / "warshall_iterations.json", warshall)

    logistic_result, _, logistic_test = stage(
        5,
        "Training Logistic Regression",
        lambda: train_logistic_regression(
            X["train"],
            y["train"],
            X["validation"],
            y["validation"],
            X["test"],
            y["test"],
            config.rare_category_threshold,
            config.random_seed,
            config.artifacts_dir / "models" / "logistic_regression.joblib",
        ),
    )
    pca_grid = config.pca_component_grid if profile == "research" else (50,)
    pca_result, _, pca_test = stage(
        6,
        "Running PCA component sensitivity and training PCA+kNN",
        lambda: train_pca_knn(
            X["train"],
            y["train"],
            X["validation"],
            y["validation"],
            X["test"],
            y["test"],
            config.rare_category_threshold,
            config.random_seed,
            config.artifacts_dir / "models" / "pca_knn.joblib",
            component_grid=pca_grid,
        ),
    )

    graph_stats, graph_sample = stage(
        7,
        "Constructing heterogeneous healthcare graph",
        lambda: build_heterogeneous_graph_summary(cleaned, config.random_seed),
    )
    LOGGER.info("      building split-local bounded cosine projections")
    projection_started = time.perf_counter()
    relation_schema = fit_relation_schema(relation_token_sets(frames["train"]))
    joblib.dump(
        relation_schema, config.artifacts_dir / "models" / "relation_schema.joblib"
    )
    projection = {
        name: build_similarity_graph(
            frames[name],
            relation_schema,
            config.graph_neighbors,
            config.graph_similarity_threshold,
        )
        for name in ("train", "validation", "test")
    }
    LOGGER.info(
        "      split-local projections completed in %.2f s",
        time.perf_counter() - projection_started,
    )
    graph_stats["encounter_projection"] = {
        name: graph.statistics for name, graph in projection.items()
    }
    graph_stats["configuration"] = {
        "graph_neighbors": config.graph_neighbors,
        "similarity_threshold": config.graph_similarity_threshold,
        "relation_vocabulary_fitted_on": "train split only",
        "split_graphs_are_disjoint": True,
    }
    write_json(config.artifacts_dir / "graphs" / "graph_statistics.json", graph_stats)
    write_json(config.artifacts_dir / "graphs" / "sample_graph.json", graph_sample)

    def prepare_gcn_features() -> tuple[dict[str, np.ndarray], Any]:
        preprocessor = make_preprocessor(config.rare_category_threshold, dense=True)
        matrices = {
            "train": np.asarray(
                preprocessor.fit_transform(X["train"]), dtype=np.float32
            )
        }
        matrices["validation"] = np.asarray(
            preprocessor.transform(X["validation"]), dtype=np.float32
        )
        matrices["test"] = np.asarray(
            preprocessor.transform(X["test"]), dtype=np.float32
        )
        joblib.dump(
            preprocessor, config.artifacts_dir / "models" / "gcn_preprocessor.joblib"
        )
        return matrices, preprocessor

    gcn_features, _ = stage(8, "Preparing fixed GCN feature schema", prepare_gcn_features)
    dataset_summary["engineered_feature_counts"] = {
        "logistic_regression": logistic_result["feature_count"],
        "pca_knn": pca_result["engineered_feature_count"],
        "gcn": int(gcn_features["train"].shape[1]),
        "graph_relations": int(len(relation_schema.feature_names_)),
    }
    adjacencies = {
        name: normalized_adjacency(
            len(graph.node_ids), graph.edges, graph.weights
        )
        for name, graph in projection.items()
    }

    primary_model, gcn_result, _, gcn_test = stage(
        9,
        "Training primary two-layer Graph Convolutional Network",
        lambda: train_gcn(
            gcn_features["train"],
            y["train"],
            adjacencies["train"],
            gcn_features["validation"],
            y["validation"],
            adjacencies["validation"],
            gcn_features["test"],
            y["test"],
            adjacencies["test"],
            hidden_size=config.gcn_hidden_size,
            dropout=config.gcn_dropout,
            learning_rate=config.gcn_learning_rate,
            weight_decay=config.gcn_weight_decay,
            maximum_epochs=config.gcn_max_epochs,
            patience=config.gcn_patience,
            random_seed=config.random_seed,
            model_path=config.artifacts_dir / "models" / "gcn_state.pt",
        ),
    )
    gcn_result["graph_statistics"] = graph_stats["encounter_projection"]

    def run_seed_stability() -> dict[str, Any]:
        seeds = (
            config.gcn_stability_seeds
            if profile == "research"
            else (config.random_seed,)
        )
        runs = [_gcn_seed_record(config.random_seed, gcn_result, True)]
        for seed in seeds:
            if seed == config.random_seed:
                continue
            LOGGER.info("      training fixed-protocol GCN seed %d", seed)
            _, seed_result, _, _ = train_gcn(
                gcn_features["train"],
                y["train"],
                adjacencies["train"],
                gcn_features["validation"],
                y["validation"],
                adjacencies["validation"],
                gcn_features["test"],
                y["test"],
                adjacencies["test"],
                hidden_size=config.gcn_hidden_size,
                dropout=config.gcn_dropout,
                learning_rate=config.gcn_learning_rate,
                weight_decay=config.gcn_weight_decay,
                maximum_epochs=config.gcn_max_epochs,
                patience=config.gcn_patience,
                random_seed=seed,
                model_path=(
                    config.artifacts_dir / "models" / f"gcn_seed_{seed}.pt"
                ),
            )
            runs.append(_gcn_seed_record(seed, seed_result, False))
        return summarize_gcn_seed_results(runs, config.random_seed)

    seed_stability = stage(10, "Quantifying GCN training-seed stability", run_seed_stability)

    robustness_seeds = (
        config.robustness_trial_seeds
        if profile == "research"
        else (config.robustness_trial_seeds[0],)
    )
    robustness_trials, robustness_summary = stage(
        11,
        "Running repeated missing/noisy relationship trials",
        lambda: run_repeated_robustness(
            primary_model,
            gcn_features["test"],
            y["test"],
            projection["test"],
            gcn_result["validation_selected_threshold"],
            config.edge_removal_levels,
            config.noise_edge_levels,
            robustness_seeds,
        ),
    )

    if profile == "research":
        ablation_results = stage(
            12,
            "Running graph-contribution ablation study",
            lambda: run_graph_ablation_study(
                primary_model,
                gcn_result,
                gcn_features,
                y,
                projection,
                hidden_size=config.gcn_hidden_size,
                dropout=config.gcn_dropout,
                learning_rate=config.gcn_learning_rate,
                weight_decay=config.gcn_weight_decay,
                maximum_epochs=config.gcn_max_epochs,
                patience=config.gcn_patience,
                random_seed=config.graph_ablation_seed,
                model_dir=config.artifacts_dir / "models",
            ),
        )
    else:
        LOGGER.info("[12/%d] Skipping graph ablation in quick mode", TOTAL_STAGES)
        ablation_results = []

    models = _comparison_rows([logistic_result, pca_result, gcn_result])
    best = max(
        models,
        key=lambda item: (
            item["test_metrics"]["pr_auc"],
            item["test_metrics"]["roc_auc"],
        ),
    )
    comparison = {
        "dataset_mode": config.dataset,
        "analysis_profile": profile,
        "ranking_basis": "Test PR-AUC, with ROC-AUC as a tie-breaker",
        "best_model": best["name"],
        "highest_recall_model": max(
            models, key=lambda item: item["test_metrics"]["recall"]
        )["name"],
        "models": models,
    }

    model_inputs = {
        "logistic_regression": {
            "name": logistic_result["name"],
            "probabilities": logistic_test,
            "threshold": logistic_result["validation_selected_threshold"],
        },
        "pca_knn": {
            "name": pca_result["name"],
            "probabilities": pca_test,
            "threshold": pca_result["validation_selected_threshold"],
        },
        "gcn": {
            "name": gcn_result["name"],
            "probabilities": gcn_test,
            "threshold": gcn_result["validation_selected_threshold"],
        },
    }
    if requested_bootstrap:
        bootstrap_confidence, paired_differences = stage(
            13,
            f"Running {requested_bootstrap} patient-clustered paired bootstrap replicates",
            lambda: patient_clustered_paired_bootstrap(
                y["test"],
                frames["test"]["patient_nbr"].astype(str).to_numpy(),
                model_inputs,
                replicates=requested_bootstrap,
                random_seed=config.bootstrap_seed,
            ),
        )
    else:
        LOGGER.info("[13/%d] Bootstrap skipped by execution profile", TOTAL_STAGES)
        skipped = {
            "metadata": {
                "status": "skipped",
                "reason": "quick mode or --skip-bootstrap",
                "requested_replicates": 0,
            }
        }
        bootstrap_confidence, paired_differences = skipped, skipped

    test_frame = frames["test"].copy()
    predictions = test_frame[["encounter_id", "patient_nbr", "target"]].copy()
    predictions = predictions.rename(columns={"target": "actual_target"})
    predictions["lr_probability"] = logistic_test
    predictions["pca_knn_probability"] = pca_test
    predictions["gcn_probability"] = gcn_test
    thresholds = {
        "lr": logistic_result["validation_selected_threshold"],
        "pca_knn": pca_result["validation_selected_threshold"],
        "gcn": gcn_result["validation_selected_threshold"],
    }
    predictions["lr_prediction"] = (logistic_test >= thresholds["lr"]).astype(int)
    predictions["pca_knn_prediction"] = (
        pca_test >= thresholds["pca_knn"]
    ).astype(int)
    predictions["gcn_prediction"] = (gcn_test >= thresholds["gcn"]).astype(int)
    predictions["split"] = "test"
    predictions["age_group"] = test_frame["age_group"].astype(str)
    predictions["admission_type"] = test_frame[
        "admission_type_description"
    ].astype(str)
    predictions["admission_source"] = test_frame[
        "admission_source_description"
    ].astype(str)
    predictions["diagnosis_categories"] = (
        test_frame[["diag_1_category", "diag_2_category", "diag_3_category"]]
        .astype(str)
        .agg("; ".join, axis=1)
    )
    predictions["active_medications"] = test_frame.apply(
        _active_medications, axis=1
    )

    evidence = neighbor_evidence(test_frame, projection["test"])
    cases = {}
    for row in predictions.to_dict(orient="records"):
        encounter = str(row["encounter_id"])
        row["encounter_id"] = encounter
        row["patient_nbr"] = str(row["patient_nbr"])
        row["nearest_graph_neighbours"] = evidence.get(encounter, [])
        cases[encounter] = row

    def save_artifacts() -> None:
        metrics_dir = config.artifacts_dir / "metrics"
        write_json(metrics_dir / "dataset_summary.json", dataset_summary)
        write_json(comparison_path, comparison)
        write_json(
            metrics_dir / "pca_component_analysis.json",
            pca_result["component_analysis"],
        )
        pd.DataFrame(
            pca_result["component_analysis"]["configurations"]
        ).to_csv(metrics_dir / "pca_component_analysis.csv", index=False)
        write_json(
            metrics_dir / "bootstrap_confidence_intervals.json",
            bootstrap_confidence,
        )
        pd.DataFrame(_bootstrap_csv_rows(bootstrap_confidence)).to_csv(
            metrics_dir / "bootstrap_confidence_intervals.csv", index=False
        )
        write_json(
            metrics_dir / "paired_model_differences.json", paired_differences
        )
        pd.DataFrame(_paired_csv_rows(paired_differences)).to_csv(
            metrics_dir / "paired_model_differences.csv", index=False
        )
        write_json(metrics_dir / "gcn_seed_stability.json", seed_stability)
        pd.DataFrame(
            [
                {
                    "seed": row["seed"],
                    "is_primary": row["is_primary_prespecified_seed"],
                    "selected_epoch": row["selected_epoch"],
                    "epochs_completed": row["epochs_completed"],
                    "selected_threshold": row["selected_threshold"],
                    "training_seconds": row["training_seconds"],
                    **{
                        metric: row["test_metrics"][metric]
                        for metric in GCN_STABILITY_METRICS
                    },
                    "confusion_matrix": row["confusion_matrix"],
                }
                for row in seed_stability["runs"]
            ]
        ).to_csv(metrics_dir / "gcn_seed_stability.csv", index=False)
        write_json(metrics_dir / "robustness_trials.json", robustness_trials)
        write_json(metrics_dir / "robustness_summary.json", robustness_summary)
        pd.DataFrame(robustness_trials).to_csv(
            metrics_dir / "robustness_trials.csv", index=False
        )
        pd.DataFrame(_robustness_summary_csv_rows(robustness_summary)).to_csv(
            metrics_dir / "robustness_summary.csv", index=False
        )
        legacy_robustness = _legacy_robustness_rows(robustness_summary)
        write_json(metrics_dir / "robustness_results.json", legacy_robustness)
        pd.DataFrame(legacy_robustness).to_csv(
            metrics_dir / "robustness_results.csv", index=False
        )
        ablation_payload = {
            "metadata": {
                "status": "complete" if ablation_results else "skipped_quick",
                "primary_seed": config.random_seed,
                "ablation_seed": config.graph_ablation_seed,
                "test_tuning_performed": False,
            },
            "results": ablation_results,
        }
        write_json(metrics_dir / "graph_ablation_results.json", ablation_payload)
        pd.DataFrame(_ablation_csv_rows(ablation_results)).to_csv(
            metrics_dir / "graph_ablation_results.csv", index=False
        )
        write_json(metrics_dir / "experiment_config.json", config.to_dict())
        predictions.to_csv(
            config.artifacts_dir / "predictions" / "test_predictions.csv",
            index=False,
        )
        write_json(
            config.artifacts_dir / "predictions" / "case_records.json", cases
        )
        pd.DataFrame(
            [
                {
                    "rank_by_test_pr_auc": model_result["rank_by_test_pr_auc"],
                    "model": model_result["name"],
                    **{
                        metric: model_result["test_metrics"][metric]
                        for metric in METRIC_KEYS
                    },
                    "threshold": model_result["validation_selected_threshold"],
                    "training_seconds": model_result["training_seconds"],
                    "inference_seconds": model_result["inference_seconds"],
                }
                for model_result in models
            ]
        ).sort_values("rank_by_test_pr_auc").to_csv(
            metrics_dir / "model_comparison.csv", index=False
        )
        _save_figures(
            models, robustness_summary, config.artifacts_dir / "figures"
        )

    stage(14, "Saving all models, analyses, predictions, and figures", save_artifacts)

    def finish() -> None:
        _integrity_check(
            config,
            comparison,
            profile=profile,
            bootstrap_replicates=requested_bootstrap,
        )
        total_seconds = time.perf_counter() - pipeline_started
        run_manifest = {
            "status": "complete",
            "dataset_mode": config.dataset,
            "analysis_profile": profile,
            "analysis_signature": signature,
            "random_seed": config.random_seed,
            "total_runtime_seconds": total_seconds,
            "models": [model_result["slug"] for model_result in models],
            "split_disjoint": dataset_summary["splits"]["disjoint"],
            "warshall_verified": warshall["verified"],
        }
        write_json(manifest_path, run_manifest)
        _write_reproducibility_manifest(
            config,
            profile,
            total_seconds,
            dataset_summary,
            requested_bootstrap,
        )

    stage(15, "Running final integrity and reproducibility checks", finish)
    final_runtime = read_json(manifest_path)["total_runtime_seconds"]
    LOGGER.info(
        "Complete in %.2f s. Best test PR-AUC: %s",
        final_runtime,
        best["name"],
    )
    return comparison


def _integrity_check(
    config: ExperimentConfig,
    comparison: dict[str, Any],
    *,
    profile: str,
    bootstrap_replicates: int,
) -> None:
    required = [
        config.split_path,
        config.artifacts_dir / "models" / "logistic_regression.joblib",
        config.artifacts_dir / "models" / "pca_knn.joblib",
        config.artifacts_dir / "models" / "gcn_state.pt",
        config.artifacts_dir / "metrics" / "robustness_trials.json",
        config.artifacts_dir / "metrics" / "robustness_summary.json",
        config.artifacts_dir / "metrics" / "gcn_seed_stability.json",
        config.artifacts_dir / "metrics" / "pca_component_analysis.json",
        config.artifacts_dir / "predictions" / "test_predictions.csv",
    ]
    if profile == "research":
        required.append(
            config.artifacts_dir / "metrics" / "graph_ablation_results.json"
        )
    if bootstrap_replicates:
        required.extend(
            [
                config.artifacts_dir
                / "metrics"
                / "bootstrap_confidence_intervals.json",
                config.artifacts_dir
                / "metrics"
                / "paired_model_differences.json",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Final artifact check failed: {missing}")
    slugs = {model["slug"] for model in comparison["models"]}
    if slugs != {"logistic_regression", "pca_knn", "gcn"}:
        raise RuntimeError(f"Comparison model set is incomplete: {slugs}")
    assignments = pd.read_csv(config.split_path, dtype=str)
    overlap = assignments.groupby("patient_nbr")["split"].nunique()
    if int((overlap > 1).sum()):
        raise RuntimeError("Final split integrity check found patient leakage")


def _print_comparison(comparison: dict[str, Any]) -> None:
    rows = sorted(
        comparison["models"], key=lambda item: item["rank_by_test_pr_auc"]
    )
    print("\nVerified test comparison")
    print("-" * 88)
    print(
        f"{'Model':34} {'PR-AUC':>9} {'ROC-AUC':>9} "
        f"{'Recall':>9} {'F1':>9} {'Seconds':>9}"
    )
    for row in rows:
        metrics = row["test_metrics"]
        print(
            f"{row['name'][:34]:34} "
            f"{metrics['pr_auc']:9.4f} {metrics['roc_auc']:9.4f} "
            f"{metrics['recall']:9.4f} {metrics['f1']:9.4f} "
            f"{row['training_seconds']:9.2f}"
        )
    print("-" * 88)
    print(f"Best by test PR-AUC: {comparison['best_model']}")
    print(f"Highest recall: {comparison['highest_recall_model']}")
    print(
        "Artifacts: artifacts/ | Split assignments: "
        "data/processed/split_assignments.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HealthGraph Readmission Lab."
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default="reduced")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--quick",
        action="store_true",
        help="Core models, one GCN seed, one robustness seed, and no default bootstrap.",
    )
    profile.add_argument(
        "--research",
        action="store_true",
        help="Full statistical, stability, robustness, PCA, and ablation analysis.",
    )
    bootstrap = parser.add_mutually_exclusive_group()
    bootstrap.add_argument(
        "--bootstrap",
        type=int,
        metavar="N",
        help="Override the patient-clustered bootstrap replicate count.",
    )
    bootstrap.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip patient-clustered bootstrap analysis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when matching complete artifacts exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    arguments = parse_args()
    execution_profile = "quick" if arguments.quick else "research"
    final_comparison = run(
        ExperimentConfig(dataset=arguments.dataset),
        force=arguments.force,
        profile=execution_profile,
        bootstrap_replicates=arguments.bootstrap,
        skip_bootstrap=arguments.skip_bootstrap,
    )
    _print_comparison(final_comparison)
