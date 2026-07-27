from pathlib import Path
from uuid import uuid4

import pytest

from app import create_app
from urop_healthgraph.artifacts import write_json
from urop_healthgraph.warshall import warshall_iterations


def fake_model(name: str, slug: str, rank: int) -> dict:
    metrics = {
        "threshold": 0.4,
        "roc_auc": 0.65,
        "pr_auc": 0.22 - rank * 0.01,
        "accuracy": 0.7,
        "balanced_accuracy": 0.6,
        "precision": 0.2,
        "recall": 0.5,
        "f1": 0.28,
        "confusion_matrix": [[8, 2], [1, 1]],
    }
    result = {
        "name": name,
        "slug": slug,
        "rank_by_test_pr_auc": rank,
        "test_metrics": metrics,
        "validation_selected_threshold": 0.4,
        "training_seconds": 1.2,
        "coefficients": {
            "strongest_positive": [{"feature": "Feature A", "coefficient": 0.3}],
            "strongest_negative": [{"feature": "Feature B", "coefficient": -0.2}],
        },
        "curves": {
            "roc": {"fpr": [0, 1], "tpr": [0, 1]},
            "precision_recall": {"recall": [0, 1], "precision": [1, 0.1]},
        },
        "history": [{"epoch": 1, "validation_pr_auc": 0.2}],
        "cumulative_explained_variance": [0.5, 0.9],
    }
    return result


@pytest.fixture
def artifact_dir() -> Path:
    tmp_path = (
        Path(__file__).resolve().parent
        / "_runtime"
        / f"web_artifacts_{uuid4().hex}"
    )
    dataset = {
        "eligible_rows": 100,
        "unique_patients": 90,
        "positive_prevalence": 0.11,
        "excluded_end_of_life_rows": 2,
        "source_columns": 50,
        "variants": {"reduced": {"rows": 100, "unique_patients": 90}},
        "splits": {
            "counts": {
                key: {
                    "encounters": count,
                    "patients": count,
                    "positive_count": 5,
                    "positive_prevalence": 0.1,
                }
                for key, count in {"train": 70, "validation": 15, "test": 15}.items()
            },
            "patient_overlap_counts": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        },
        "distributions": {
            "age": [{"label": "[60-70)", "count": 30}],
            "primary_diagnosis": [{"label": "Diabetes", "count": 20}],
            "active_medications": [{"label": "insulin", "count": 30}],
            "readmission": [{"label": "Positive", "count": 11}],
        },
        "initial_missingness": {"weight": 0.96},
        "removed_columns": [{"column": "weight", "reason": "Missing"}],
        "data_dictionary": {
            "columns": [
                {"column": "age", "role": "Predictor source", "description": "Age"}
            ],
            "id_mappings": {"admission_type_id": {"1": "Emergency"}},
        },
    }
    models = [
        fake_model("Logistic Regression", "logistic_regression", 1),
        fake_model("PCA + k-Nearest Neighbours", "pca_knn", 2),
        fake_model("Graph Convolutional Network", "gcn", 3),
    ]
    graph_stats = {
        "total_nodes": 200,
        "total_edges": 400,
        "representation_note": "Sampled only.",
        "encounter_projection": {"test": {"undirected_edges": 50}},
    }
    graph = {
        "nodes": [
            {
                "id": "E:1",
                "type": "Encounter",
                "label": "Encounter 1",
                "encounter_id": "1",
            }
        ],
        "links": [],
        "metadata": {"1": {}},
        "sample_encounters": 1,
    }
    robustness_trials = [
        {
            "scenario": "baseline",
            "label": "Baseline",
            "perturbation_type": "baseline",
            "trial_seed": 42,
            "edge_count_after": 50,
            "pr_auc": 0.2,
            "roc_auc": 0.6,
            "recall": 0.5,
            "f1": 0.25,
            "balanced_accuracy": 0.55,
            "probability_shift_mean_absolute": 0.0,
        }
    ]
    robustness_summary = [
        {
            "scenario": "baseline",
            "label": "Baseline",
            "perturbation_type": "baseline",
            "perturbation_level": 0,
            "number_of_trials": 1,
            "trial_seeds": [42],
            "edge_count_before": 50,
            "edge_count_after": {"mean": 50},
            "metrics": {
                metric: {
                    "n": 1,
                    "mean": value,
                    "standard_deviation": 0,
                    "median": value,
                    "minimum": value,
                    "maximum": value,
                    "ci_95_lower": value,
                    "ci_95_upper": value,
                }
                for metric, value in {
                    "pr_auc": 0.2,
                    "roc_auc": 0.6,
                    "balanced_accuracy": 0.55,
                    "recall": 0.5,
                    "f1": 0.25,
                    "probability_shift_mean_absolute": 0,
                }.items()
            },
        }
    ]
    bootstrap = {
        "metadata": {
            "valid_replicates": 20,
            "invalid_single_class_replicates": 0,
        },
        "models": {
            model["slug"]: {
                "name": model["name"],
                "intervals": {
                    metric: {
                        "estimate": model["test_metrics"][metric],
                        "bootstrap_mean": model["test_metrics"][metric],
                        "ci_95_lower": max(0, model["test_metrics"][metric] - 0.02),
                        "ci_95_upper": min(1, model["test_metrics"][metric] + 0.02),
                    }
                    for metric in (
                        "pr_auc",
                        "roc_auc",
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
            }
            for model in models
        },
    }
    paired = {
        "metadata": {"probability_note": "Descriptive paired-resampling measure."},
        "comparisons": [
            {
                "left_name": "Logistic Regression",
                "right_name": "Graph Convolutional Network",
                "metrics": {
                    metric: {
                        "observed_difference": 0.01,
                        "bootstrap_mean_difference": 0.01,
                        "ci_95_lower": -0.01,
                        "ci_95_upper": 0.03,
                        "proportion_above_zero": 0.7,
                        "two_sided_bootstrap_tail_probability": 0.4,
                    }
                    for metric in (
                        "pr_auc",
                        "roc_auc",
                        "recall",
                        "f1",
                    )
                },
            }
        ],
    }
    seed_stability = {
        "metadata": {
            "primary_seed": 42,
            "number_of_seeds": 1,
            "selection_policy": "Primary seed retained.",
        },
        "runs": [
            {
                "seed": 42,
                "is_primary_prespecified_seed": True,
                "selected_epoch": 2,
                "selected_threshold": 0.4,
                "test_metrics": models[2]["test_metrics"],
            }
        ],
        "summary": {},
    }
    pca_analysis = {
        "selected_components": 25,
        "selected_neighbors": 5,
        "selection_criterion": "Validation PR-AUC.",
        "configurations": [
            {
                "status": "completed",
                "components": 25,
                "explained_variance": 0.6,
                "validation_pr_auc": 0.2,
                "runtime_seconds": 1,
            }
        ],
    }
    ablation = {
        "metadata": {"status": "complete"},
        "results": [
            {
                "label": "Original graph",
                "training_strategy": "Primary model.",
                "test_metrics": models[2]["test_metrics"],
            }
        ],
    }
    case = {
        "1": {
            "encounter_id": "1",
            "patient_nbr": "10",
            "actual_target": 0,
            "lr_probability": 0.2,
            "pca_knn_probability": 0.3,
            "gcn_probability": 0.4,
            "lr_prediction": 0,
            "pca_knn_prediction": 0,
            "gcn_prediction": 1,
            "nearest_graph_neighbours": [],
        }
    }
    payloads = {
        "metrics/dataset_summary.json": dataset,
        "metrics/model_comparison.json": {
            "models": models,
            "best_model": "Logistic Regression",
            "highest_recall_model": "Graph Convolutional Network",
            "ranking_basis": "Test PR-AUC.",
        },
        "metrics/bootstrap_confidence_intervals.json": bootstrap,
        "metrics/paired_model_differences.json": paired,
        "metrics/gcn_seed_stability.json": seed_stability,
        "metrics/pca_component_analysis.json": pca_analysis,
        "metrics/graph_ablation_results.json": ablation,
        "graphs/warshall_iterations.json": warshall_iterations(),
        "graphs/sample_graph.json": graph,
        "graphs/graph_statistics.json": graph_stats,
        "metrics/robustness_trials.json": robustness_trials,
        "metrics/robustness_summary.json": robustness_summary,
        "predictions/case_records.json": case,
        "metrics/run_manifest.json": {"status": "complete"},
    }
    for relative, payload in payloads.items():
        write_json(tmp_path / relative, payload)
    return tmp_path


def test_flask_route_smoke_tests(artifact_dir: Path) -> None:
    app = create_app({"TESTING": True, "ARTIFACTS_DIR": artifact_dir})
    client = app.test_client()
    for route in (
        "/",
        "/dataset",
        "/warshall",
        "/models",
        "/graph",
        "/robustness",
        "/cases",
        "/limitations",
    ):
        response = client.get(route)
        assert response.status_code == 200, route
        assert b"HealthGraph" in response.data
    assert client.get("/api/health").json["ready"] is True
    assert client.get("/api/cases/1").json["encounter_id"] == "1"
    assert client.get("/api/statistics").status_code == 200
    assert client.get("/api/robustness").json["summary"][0]["scenario"] == "baseline"
    assert client.get("/api/pca-analysis").status_code == 200
    assert client.get("/api/ablation").status_code == 200


def test_downloadable_result_controls(artifact_dir: Path) -> None:
    metrics = artifact_dir / "metrics"
    (metrics / "bootstrap_confidence_intervals.csv").write_text(
        "model,metric\nlr,pr_auc\n", encoding="utf-8"
    )
    app = create_app({"TESTING": True, "ARTIFACTS_DIR": artifact_dir})
    client = app.test_client()
    response = client.get(
        "/downloads/results/bootstrap_confidence_intervals.csv"
    )
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment")
    assert client.get("/downloads/results/not-allowed.txt").status_code == 404


def test_missing_artifact_error_behavior() -> None:
    empty_path = (
        Path(__file__).resolve().parent
        / "_runtime"
        / f"missing_artifacts_{uuid4().hex}"
    )
    empty_path.mkdir(parents=True, exist_ok=True)
    app = create_app({"TESTING": True, "ARTIFACTS_DIR": empty_path})
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"python run_experiments.py --dataset reduced" in page.data
    api = client.get("/api/dataset")
    assert api.status_code == 503
    assert api.json["status"] == "artifacts_missing"
    assert client.get("/api/health").json["ready"] is False
