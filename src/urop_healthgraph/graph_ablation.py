"""GCN graph-contribution ablations with validation-only model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import classification_metrics
from .gcn_model import TwoLayerGCN, predict_gcn, train_gcn
from .graph_builder import GraphProjection, normalized_adjacency


ABLATION_METRICS = (
    "pr_auc",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def random_simple_graph(
    node_count: int,
    edge_count: int,
    reference_weights: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an undirected simple random graph with an exact edge count."""
    possible = node_count * (node_count - 1) // 2
    if edge_count > possible:
        raise ValueError("Requested more edges than a simple graph can contain")
    if edge_count == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    rng = np.random.default_rng(random_seed)
    selected: set[tuple[int, int]] = set()
    while len(selected) < edge_count:
        remaining = edge_count - len(selected)
        candidates = rng.integers(0, node_count, size=(max(remaining * 2, 256), 2))
        for left, right in candidates:
            if left == right:
                continue
            pair = (int(min(left, right)), int(max(left, right)))
            selected.add(pair)
            if len(selected) == edge_count:
                break
    edges = np.asarray(sorted(selected), dtype=np.int64)
    if len(reference_weights):
        weights = rng.choice(reference_weights, size=edge_count, replace=True).astype(
            np.float32
        )
    else:
        weights = np.ones(edge_count, dtype=np.float32)
    return edges, weights


def _metric_record(
    scenario: str,
    label: str,
    design: str,
    training_strategy: str,
    threshold: float,
    metrics: dict[str, Any],
    *,
    training_seconds: float,
    test_edges: int,
    selected_epoch: int | None,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "label": label,
        "design": design,
        "training_strategy": training_strategy,
        "threshold": float(threshold),
        "training_seconds": float(training_seconds),
        "selected_epoch": selected_epoch,
        "test_edge_count": int(test_edges),
        "test_metrics": metrics,
    }


def run_graph_ablation_study(
    primary_model: TwoLayerGCN,
    primary_result: dict[str, Any],
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    projections: dict[str, GraphProjection],
    *,
    hidden_size: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    random_seed: int,
    model_dir: Path,
) -> list[dict[str, Any]]:
    """Evaluate original, identity, random, and feature-alignment conditions.

    Identity and random graph conditions are retrained using only their
    corresponding train and validation graphs. Feature shuffling is an
    inference-only alignment stress test of the prespecified primary GCN.
    """
    results = [
        _metric_record(
            "original_graph",
            "Original clinical similarity graph",
            "Clinically constructed split-local relation graph.",
            "Primary prespecified GCN trained on the original train graph.",
            primary_result["validation_selected_threshold"],
            primary_result["test_metrics"],
            training_seconds=primary_result["training_seconds"],
            test_edges=len(projections["test"].edges),
            selected_epoch=primary_result["best_epoch"],
        )
    ]

    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_weights = np.empty(0, dtype=np.float32)
    identity_adjacency = {
        name: normalized_adjacency(len(graph.node_ids), empty_edges, empty_weights)
        for name, graph in projections.items()
    }
    _, identity_result, _, _ = train_gcn(
        features["train"],
        labels["train"],
        identity_adjacency["train"],
        features["validation"],
        labels["validation"],
        identity_adjacency["validation"],
        features["test"],
        labels["test"],
        identity_adjacency["test"],
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        maximum_epochs=maximum_epochs,
        patience=patience,
        random_seed=random_seed + 1_000,
        model_path=model_dir / "gcn_ablation_identity.pt",
    )
    results.append(
        _metric_record(
            "identity_only",
            "Identity-only adjacency",
            "Self-loops only; encounters cannot aggregate neighbour information.",
            "Retrained on identity-only train adjacency; epoch and threshold selected on identity-only validation adjacency.",
            identity_result["validation_selected_threshold"],
            identity_result["test_metrics"],
            training_seconds=identity_result["training_seconds"],
            test_edges=0,
            selected_epoch=identity_result["best_epoch"],
        )
    )

    random_graphs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    random_adjacency = {}
    for offset, (name, graph) in enumerate(projections.items()):
        edges, weights = random_simple_graph(
            len(graph.node_ids),
            len(graph.edges),
            graph.weights,
            random_seed + 2_000 + offset,
        )
        random_graphs[name] = (edges, weights)
        random_adjacency[name] = normalized_adjacency(
            len(graph.node_ids), edges, weights
        )
    _, random_result, _, _ = train_gcn(
        features["train"],
        labels["train"],
        random_adjacency["train"],
        features["validation"],
        labels["validation"],
        random_adjacency["validation"],
        features["test"],
        labels["test"],
        random_adjacency["test"],
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        maximum_epochs=maximum_epochs,
        patience=patience,
        random_seed=random_seed + 2_000,
        model_path=model_dir / "gcn_ablation_random.pt",
    )
    results.append(
        _metric_record(
            "random_graph",
            "Matched random graph",
            "Uniform simple random graph with the original node and edge counts and resampled edge weights.",
            "Retrained on random train adjacency; epoch and threshold selected on random validation adjacency.",
            random_result["validation_selected_threshold"],
            random_result["test_metrics"],
            training_seconds=random_result["training_seconds"],
            test_edges=len(random_graphs["test"][0]),
            selected_epoch=random_result["best_epoch"],
        )
    )

    rng = np.random.default_rng(random_seed + 3_000)
    permutation = rng.permutation(len(features["test"]))
    shuffled_probability = predict_gcn(
        primary_model,
        features["test"][permutation],
        normalized_adjacency(
            len(projections["test"].node_ids),
            projections["test"].edges,
            projections["test"].weights,
        ),
    )
    shuffled_metrics = classification_metrics(
        labels["test"],
        shuffled_probability,
        primary_result["validation_selected_threshold"],
    )
    results.append(
        _metric_record(
            "feature_shuffled",
            "Feature-shuffled inference",
            "Test feature rows are deterministically permuted while graph edges and labels remain fixed.",
            "Inference-only stress test of the primary GCN; no retraining and no test-set tuning.",
            primary_result["validation_selected_threshold"],
            shuffled_metrics,
            training_seconds=0.0,
            test_edges=len(projections["test"].edges),
            selected_epoch=primary_result["best_epoch"],
        )
    )

    original = results[0]["test_metrics"]
    for result in results:
        result["difference_from_original"] = {
            metric: float(result["test_metrics"][metric] - original[metric])
            for metric in ABLATION_METRICS
        }
    return results
