"""Reproducible perturbations of graph relationships for GCN robustness."""

from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation import classification_metrics
from .gcn_model import TwoLayerGCN, predict_gcn
from .graph_builder import GraphProjection, normalized_adjacency
from .statistical_analysis import descriptive_summary


ROBUSTNESS_METRICS = (
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "recall",
    "f1",
    "probability_shift_mean_absolute",
)


def remove_random_edges(
    graph: GraphProjection, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a fixed fraction of undirected edges without changing features."""
    rng = np.random.default_rng(seed)
    remove_count = int(round(len(graph.edges) * fraction))
    keep = np.ones(len(graph.edges), dtype=bool)
    if remove_count:
        keep[rng.choice(len(graph.edges), size=remove_count, replace=False)] = False
    return graph.edges[keep], graph.weights[keep]


def add_random_noise_edges(
    graph: GraphProjection, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Add non-self random edges not already present in the encounter graph."""
    rng = np.random.default_rng(seed)
    requested = int(round(len(graph.edges) * fraction))
    existing = {tuple(edge) for edge in graph.edges.tolist()}
    added: set[tuple[int, int]] = set()
    attempts = 0
    maximum_attempts = max(100, requested * 30)
    node_count = len(graph.node_ids)
    while len(added) < requested and attempts < maximum_attempts:
        left, right = rng.integers(0, node_count, size=2)
        attempts += 1
        if left == right:
            continue
        pair = (int(min(left, right)), int(max(left, right)))
        if pair not in existing:
            added.add(pair)
    if added:
        added_array = np.asarray(sorted(added), dtype=np.int64)
        edges = np.vstack([graph.edges, added_array])
        baseline_weight = (
            float(np.median(graph.weights)) if len(graph.weights) else 1.0
        )
        weights = np.concatenate(
            [
                graph.weights,
                np.full(len(added_array), baseline_weight, dtype=np.float32),
            ]
        )
        return edges, weights
    return graph.edges.copy(), graph.weights.copy()


def run_robustness_simulation(
    model: TwoLayerGCN,
    features: np.ndarray,
    labels: np.ndarray,
    graph: GraphProjection,
    threshold: float,
    removal_levels: tuple[float, ...],
    noise_levels: tuple[float, ...],
    random_seed: int,
) -> list[dict[str, Any]]:
    """Evaluate relational incompleteness and erroneous-edge scenarios."""
    baseline_adjacency = normalized_adjacency(
        len(graph.node_ids), graph.edges, graph.weights
    )
    baseline_probability = predict_gcn(model, features, baseline_adjacency)
    scenarios: list[tuple[str, str, float, np.ndarray, np.ndarray]] = [
        ("baseline", "Baseline", 0.0, graph.edges, graph.weights)
    ]
    for index, level in enumerate(removal_levels, start=1):
        edges, weights = remove_random_edges(graph, level, random_seed + index)
        scenarios.append(
            (
                f"remove_{int(level * 100)}",
                f"Remove {int(level * 100)}% edges",
                level,
                edges,
                weights,
            )
        )
    for index, level in enumerate(noise_levels, start=20):
        edges, weights = add_random_noise_edges(graph, level, random_seed + index)
        scenarios.append(
            (
                f"noise_{int(level * 100)}",
                f"Add {int(level * 100)}% noise edges",
                level,
                edges,
                weights,
            )
        )

    results: list[dict[str, Any]] = []
    for slug, label, level, edges, weights in scenarios:
        adjacency = normalized_adjacency(len(graph.node_ids), edges, weights)
        probability = (
            baseline_probability
            if slug == "baseline"
            else predict_gcn(model, features, adjacency)
        )
        metrics = classification_metrics(labels, probability, threshold)
        results.append(
            {
                "scenario": slug,
                "label": label,
                "perturbation_level": float(level),
                "edge_count_before": int(len(graph.edges)),
                "edge_count_after": int(len(edges)),
                "probability_shift_mean_absolute": float(
                    np.mean(np.abs(probability - baseline_probability))
                ),
                **metrics,
            }
        )
    return results


def run_repeated_robustness(
    model: TwoLayerGCN,
    features: np.ndarray,
    labels: np.ndarray,
    graph: GraphProjection,
    threshold: float,
    removal_levels: tuple[float, ...],
    noise_levels: tuple[float, ...],
    trial_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every graph-relationship perturbation over common deterministic seeds."""
    if not trial_seeds:
        raise ValueError("At least one robustness trial seed is required")
    baseline_adjacency = normalized_adjacency(
        len(graph.node_ids), graph.edges, graph.weights
    )
    baseline_probability = predict_gcn(model, features, baseline_adjacency)
    scenarios: list[tuple[str, str, str, float]] = [
        ("baseline", "Baseline graph", "baseline", 0.0)
    ]
    scenarios.extend(
        (
            f"remove_{int(level * 100)}",
            f"Remove {int(level * 100)}% edges",
            "edge_removal",
            level,
        )
        for level in removal_levels
    )
    scenarios.extend(
        (
            f"noise_{int(level * 100)}",
            f"Add {int(level * 100)}% noise edges",
            "noise_addition",
            level,
        )
        for level in noise_levels
    )

    trials: list[dict[str, Any]] = []
    for scenario_index, (slug, label, perturbation_type, level) in enumerate(scenarios):
        for seed in trial_seeds:
            derived_seed = int(seed + scenario_index * 10_000)
            if perturbation_type == "baseline":
                edges = graph.edges
                weights = graph.weights
                probability = baseline_probability
            elif perturbation_type == "edge_removal":
                edges, weights = remove_random_edges(graph, level, derived_seed)
                probability = predict_gcn(
                    model,
                    features,
                    normalized_adjacency(len(graph.node_ids), edges, weights),
                )
            else:
                edges, weights = add_random_noise_edges(graph, level, derived_seed)
                probability = predict_gcn(
                    model,
                    features,
                    normalized_adjacency(len(graph.node_ids), edges, weights),
                )
            if len(edges):
                if np.any(edges[:, 0] == edges[:, 1]):
                    raise AssertionError("Robustness perturbation created a self-edge")
                if len({tuple(edge) for edge in edges.tolist()}) != len(edges):
                    raise AssertionError("Robustness perturbation created duplicate edges")
            metrics = classification_metrics(labels, probability, threshold)
            trials.append(
                {
                    "scenario": slug,
                    "label": label,
                    "perturbation_type": perturbation_type,
                    "perturbation_level": float(level),
                    "trial_seed": int(seed),
                    "derived_seed": derived_seed,
                    "node_count": int(len(graph.node_ids)),
                    "edge_count_before": int(len(graph.edges)),
                    "edge_count_after": int(len(edges)),
                    "probability_shift_mean_absolute": float(
                        np.mean(np.abs(probability - baseline_probability))
                    ),
                    **metrics,
                }
            )

    summaries: list[dict[str, Any]] = []
    for slug, label, perturbation_type, level in scenarios:
        selected = [row for row in trials if row["scenario"] == slug]
        summary = {
            "scenario": slug,
            "label": label,
            "perturbation_type": perturbation_type,
            "perturbation_level": float(level),
            "number_of_trials": len(selected),
            "trial_seeds": [int(row["trial_seed"]) for row in selected],
            "edge_count_before": int(len(graph.edges)),
            "edge_count_after": descriptive_summary(
                row["edge_count_after"] for row in selected
            ),
            "metrics": {
                metric: descriptive_summary(row[metric] for row in selected)
                for metric in ROBUSTNESS_METRICS
            },
        }
        summaries.append(summary)
    return trials, summaries
