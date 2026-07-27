"""Logistic Regression and PCA+kNN experiment implementations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification_metrics, curve_points, select_threshold
from .feature_engineering import (
    make_preprocessor,
    readable_feature_name,
)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    rare_threshold: float,
    random_seed: int,
    model_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Tune on validation data, evaluate once on test data, and serialize."""
    started = time.perf_counter()
    candidates: list[tuple[float, float, Pipeline, np.ndarray]] = []
    for c_value in (0.1, 1.0):
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(rare_threshold, dense=False)),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=500,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        score = classification_metrics(y_validation, probabilities, 0.5)["pr_auc"]
        candidates.append((float(score or 0.0), c_value, pipeline, probabilities))

    _, best_c, best_pipeline, validation_probabilities = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    threshold = select_threshold(y_validation, validation_probabilities)
    training_seconds = time.perf_counter() - started

    inference_started = time.perf_counter()
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - inference_started

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    ranked = sorted(
        [
            {
                "feature": readable_feature_name(str(name)),
                "coefficient": float(value),
                "odds_ratio": float(np.exp(np.clip(value, -20, 20))),
            }
            for name, value in zip(names, coefficients, strict=True)
        ],
        key=lambda item: item["coefficient"],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    result = {
        "name": "Logistic Regression",
        "slug": "logistic_regression",
        "selected_hyperparameters": {"C": best_c, "class_weight": "balanced"},
        "validation_selected_threshold": threshold,
        "validation_metrics": classification_metrics(
            y_validation, validation_probabilities, threshold
        ),
        "test_metrics": classification_metrics(y_test, test_probabilities, threshold),
        "test_metrics_at_0_5": classification_metrics(
            y_test, test_probabilities, 0.5
        ),
        "curves": curve_points(y_test, test_probabilities),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "feature_count": int(len(names)),
        "coefficients": {
            "strongest_negative": ranked[:12],
            "strongest_positive": list(reversed(ranked[-12:])),
        },
    }
    return result, validation_probabilities, test_probabilities


def train_pca_knn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    rare_threshold: float,
    random_seed: int,
    model_path: Path,
    component_grid: tuple[int, ...] = (25, 50, 75, 100, 150),
    neighbor_grid: tuple[int, ...] = (7, 15, 31),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Select PCA components and distance-weighted kNN using validation only."""
    started = time.perf_counter()
    preprocessor = make_preprocessor(rare_threshold, dense=True)
    train_unscaled = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    validation_unscaled = np.asarray(
        preprocessor.transform(X_validation), dtype=np.float32
    )
    test_unscaled = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    matrix_scaler = StandardScaler()
    train_engineered = matrix_scaler.fit_transform(train_unscaled).astype(
        np.float32, copy=False
    )
    validation_engineered = matrix_scaler.transform(validation_unscaled).astype(
        np.float32, copy=False
    )
    test_engineered = matrix_scaler.transform(test_unscaled).astype(
        np.float32, copy=False
    )

    maximum_components = min(
        train_engineered.shape[0] - 1, train_engineered.shape[1]
    )
    allowed_components = sorted(
        {int(value) for value in component_grid if 2 <= value <= maximum_components}
    )
    skipped_components = sorted(
        {int(value) for value in component_grid if value not in allowed_components}
    )
    if not allowed_components:
        allowed_components = [maximum_components]

    component_records: list[dict[str, Any]] = []
    fitted_candidates: list[dict[str, Any]] = []
    for components in allowed_components:
        component_started = time.perf_counter()
        try:
            pca_candidate = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=random_seed,
            )
            train_reduced = pca_candidate.fit_transform(train_engineered)
            validation_reduced = pca_candidate.transform(validation_engineered)
            neighbor_candidates = []
            for neighbors in neighbor_grid:
                model_candidate = KNeighborsClassifier(
                    n_neighbors=neighbors, weights="distance", n_jobs=1
                )
                model_candidate.fit(train_reduced, y_train)
                probabilities = model_candidate.predict_proba(validation_reduced)[:, 1]
                metrics = classification_metrics(y_validation, probabilities, 0.5)
                neighbor_candidates.append(
                    {
                        "pr_auc": float(metrics["pr_auc"] or 0.0),
                        "roc_auc": float(metrics["roc_auc"] or 0.0),
                        "neighbors": int(neighbors),
                        "model": model_candidate,
                        "probabilities": probabilities,
                    }
                )
            best_neighbor = max(
                neighbor_candidates,
                key=lambda item: (
                    item["pr_auc"],
                    item["roc_auc"],
                    -item["neighbors"],
                ),
            )
            validation_threshold = select_threshold(
                y_validation, best_neighbor["probabilities"]
            )
            validation_metrics = classification_metrics(
                y_validation,
                best_neighbor["probabilities"],
                validation_threshold,
            )
            runtime = time.perf_counter() - component_started
            record = {
                "status": "completed",
                "components": int(components),
                "explained_variance": float(
                    np.sum(pca_candidate.explained_variance_ratio_)
                ),
                "selected_neighbors": best_neighbor["neighbors"],
                "validation_threshold": validation_threshold,
                "validation_pr_auc": float(validation_metrics["pr_auc"]),
                "validation_roc_auc": float(validation_metrics["roc_auc"]),
                "validation_balanced_accuracy": float(
                    validation_metrics["balanced_accuracy"]
                ),
                "validation_recall": float(validation_metrics["recall"]),
                "validation_f1": float(validation_metrics["f1"]),
                "runtime_seconds": float(runtime),
            }
            component_records.append(record)
            fitted_candidates.append(
                {
                    **record,
                    "pca": pca_candidate,
                    "model": best_neighbor["model"],
                    "probabilities": best_neighbor["probabilities"],
                }
            )
        except MemoryError as error:
            component_records.append(
                {
                    "status": "stopped_memory_limit",
                    "components": int(components),
                    "runtime_seconds": float(time.perf_counter() - component_started),
                    "error": str(error),
                }
            )
            break
    if not fitted_candidates:
        raise RuntimeError("No PCA component configuration completed")

    selected = max(
        fitted_candidates,
        key=lambda item: (
            item["validation_pr_auc"],
            item["validation_roc_auc"],
            -item["components"],
        ),
    )
    pca = selected["pca"]
    best_model = selected["model"]
    validation_probabilities = selected["probabilities"]
    selected_components = int(selected["components"])
    best_neighbors = int(selected["selected_neighbors"])
    threshold = select_threshold(y_validation, validation_probabilities)
    training_seconds = time.perf_counter() - started

    inference_started = time.perf_counter()
    test_reduced = pca.transform(test_engineered)
    test_probabilities = best_model.predict_proba(test_reduced)[:, 1]
    inference_seconds = time.perf_counter() - inference_started

    fitted_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("matrix_scaler", matrix_scaler),
            ("pca", pca),
            ("knn", best_model),
        ]
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, model_path)
    result = {
        "name": "PCA + k-Nearest Neighbours",
        "slug": "pca_knn",
        "selected_hyperparameters": {
            "n_neighbors": best_neighbors,
            "weights": "distance",
            "pca_components": selected_components,
            "pca_component_grid_maximum_allowed": maximum_components,
            "pca_achieved_variance": float(np.sum(pca.explained_variance_ratio_)),
            "validation_selection_criterion": (
                "Highest validation PR-AUC; validation ROC-AUC tie-break; "
                "then fewer components."
            ),
        },
        "validation_selected_threshold": threshold,
        "validation_metrics": classification_metrics(
            y_validation, validation_probabilities, threshold
        ),
        "test_metrics": classification_metrics(y_test, test_probabilities, threshold),
        "test_metrics_at_0_5": classification_metrics(
            y_test, test_probabilities, 0.5
        ),
        "curves": curve_points(y_test, test_probabilities),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "engineered_feature_count": int(train_engineered.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(
            pca.explained_variance_ratio_
        ).tolist(),
        "component_analysis": {
            "requested_grid": [int(value) for value in component_grid],
            "allowed_maximum_components": int(maximum_components),
            "skipped_invalid_components": skipped_components,
            "selection_uses_test_data": False,
            "selection_criterion": (
                "Highest validation PR-AUC; validation ROC-AUC tie-break; "
                "then fewer components."
            ),
            "selected_components": selected_components,
            "selected_neighbors": best_neighbors,
            "configurations": component_records,
        },
    }
    return result, validation_probabilities, test_probabilities
