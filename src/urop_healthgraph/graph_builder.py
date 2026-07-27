"""Heterogeneous graph summaries and scalable encounter similarity graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from bisect import bisect_left
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from .feature_engineering import MEDICATION_COLUMNS


@dataclass(slots=True)
class GraphProjection:
    """An undirected weighted encounter graph without normalization self-loops."""

    node_ids: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    relation_tokens: list[set[str]]
    statistics: dict[str, Any]


def _node(identifier: str, node_type: str, label: str, **extra: Any) -> dict[str, Any]:
    return {"id": identifier, "type": node_type, "label": label, **extra}


def build_heterogeneous_graph_summary(
    frame: pd.DataFrame, random_seed: int = 42, sample_encounters: int = 55
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Count the complete heterogeneous representation and build a safe sample."""
    node_sets: dict[str, set[str]] = defaultdict(set)
    edge_counts: Counter[str] = Counter()
    degrees: Counter[str] = Counter()
    medication_nodes: set[str] = set()
    lab_nodes: set[str] = set()

    for row in frame.to_dict(orient="records"):
        encounter = f"E:{row['encounter_id']}"
        patient = f"P:{row['patient_nbr']}"
        node_sets["Encounter"].add(encounter)
        node_sets["Patient"].add(patient)
        edge_counts["has_encounter"] += 1
        degrees[encounter] += 1
        degrees[patient] += 1
        for category in {
            row["diag_1_category"],
            row["diag_2_category"],
            row["diag_3_category"],
        } - {"Unknown"}:
            diagnosis = f"D:{category}"
            node_sets["DiagnosisCategory"].add(diagnosis)
            edge_counts["diagnosed_with"] += 1
            degrees[encounter] += 1
            degrees[diagnosis] += 1
        for medication in MEDICATION_COLUMNS:
            if row.get(f"med_{medication}_active", 0):
                medication_node = f"M:{medication}"
                medication_nodes.add(medication_node)
                edge_counts["received_medication"] += 1
                degrees[encounter] += 1
                degrees[medication_node] += 1
        admission_type = f"AT:{row['admission_type_description']}"
        admission_source = f"AS:{row['admission_source_description']}"
        node_sets["AdmissionType"].add(admission_type)
        node_sets["AdmissionSource"].add(admission_source)
        for relation, target in (
            ("admitted_as", admission_type),
            ("originated_from", admission_source),
        ):
            edge_counts[relation] += 1
            degrees[encounter] += 1
            degrees[target] += 1
        for lab_name, lab_value in (
            ("A1C", row["A1Cresult"]),
            ("Glucose", row["max_glu_serum"]),
        ):
            if lab_value != "Not Measured":
                lab = f"L:{lab_name}:{lab_value}"
                lab_nodes.add(lab)
                edge_counts["has_lab_result"] += 1
                degrees[encounter] += 1
                degrees[lab] += 1

    node_sets["Medication"] = medication_nodes
    node_sets["LabResultCategory"] = lab_nodes
    patient_sizes = frame.groupby("patient_nbr").size()
    degree_values = np.asarray(list(degrees.values()), dtype=float)
    stats = {
        "node_counts": {key: len(value) for key, value in node_sets.items()},
        "edge_counts": dict(edge_counts),
        "total_nodes": int(sum(len(value) for value in node_sets.values())),
        "total_edges": int(sum(edge_counts.values())),
        "degree_summary": {
            "minimum": float(degree_values.min()) if len(degree_values) else 0.0,
            "mean": float(degree_values.mean()) if len(degree_values) else 0.0,
            "median": float(np.median(degree_values)) if len(degree_values) else 0.0,
            "maximum": float(degree_values.max(initial=0)),
        },
        "patient_encounter_components": {
            "components": int(frame["patient_nbr"].nunique()),
            "largest_component_nodes": int(patient_sizes.max() + 1),
            "interpretation": "Each component contains one patient and their encounter nodes.",
        },
        "representation_note": (
            "Counts use a tabular edge representation; NetworkX is restricted to "
            "the sampled browser graph to avoid wasteful full-graph materialization."
        ),
    }

    sample_size = min(sample_encounters, len(frame))
    sample = frame.sample(sample_size, random_state=random_seed)
    graph = nx.Graph()
    encounter_metadata: dict[str, dict[str, Any]] = {}
    for row in sample.to_dict(orient="records"):
        encounter = f"E:{row['encounter_id']}"
        patient = f"P:{row['patient_nbr']}"
        graph.add_node(
            encounter,
            **_node(
                encounter,
                "Encounter",
                f"Encounter {row['encounter_id']}",
                encounter_id=str(row["encounter_id"]),
                age_group=str(row["age_group"]),
                target=int(row["target"]),
                outcome="Readmitted <30 days" if row["target"] else "Not within 30 days",
            ),
        )
        graph.add_node(patient, **_node(patient, "Patient", "Anonymous patient"))
        graph.add_edge(patient, encounter, relation="has_encounter")
        diagnoses: list[str] = []
        medications: list[str] = []
        for category in {
            row["diag_1_category"],
            row["diag_2_category"],
            row["diag_3_category"],
        } - {"Unknown"}:
            target = f"D:{category}"
            diagnoses.append(str(category))
            graph.add_node(target, **_node(target, "DiagnosisCategory", str(category)))
            graph.add_edge(encounter, target, relation="diagnosed_with")
        for medication in MEDICATION_COLUMNS:
            if row.get(f"med_{medication}_active", 0):
                target = f"M:{medication}"
                medications.append(medication)
                graph.add_node(target, **_node(target, "Medication", medication))
                graph.add_edge(encounter, target, relation="received_medication")
        for prefix, kind, relation, label in (
            ("AT", "AdmissionType", "admitted_as", row["admission_type_description"]),
            ("AS", "AdmissionSource", "originated_from", row["admission_source_description"]),
        ):
            target = f"{prefix}:{label}"
            graph.add_node(target, **_node(target, kind, str(label)))
            graph.add_edge(encounter, target, relation=relation)
        encounter_metadata[str(row["encounter_id"])] = {
            "diagnoses": diagnoses,
            "active_medications": medications,
            "admission_type": str(row["admission_type_description"]),
            "admission_source": str(row["admission_source_description"]),
        }

    sample_json = {
        "nodes": [attributes for _, attributes in graph.nodes(data=True)],
        "links": [
            {"source": source, "target": target, **attributes}
            for source, target, attributes in graph.edges(data=True)
        ],
        "metadata": encounter_metadata,
        "sample_encounters": sample_size,
        "full_graph_rendered": False,
    }
    return stats, sample_json


def relation_token_sets(frame: pd.DataFrame) -> list[set[str]]:
    """Create relation evidence tokens used for encounter similarity."""
    result: list[set[str]] = []
    for row in frame.to_dict(orient="records"):
        tokens = {
            f"diagnosis:{row['diag_1_category']}",
            f"diagnosis:{row['diag_2_category']}",
            f"diagnosis:{row['diag_3_category']}",
            f"admission_type:{row['admission_type_description']}",
            f"admission_source:{row['admission_source_description']}",
        }
        tokens.discard("diagnosis:Unknown")
        for medication in MEDICATION_COLUMNS:
            if row.get(f"med_{medication}_active", 0):
                tokens.add(f"medication:{medication}")
        result.append(tokens)
    return result


def fit_relation_schema(token_sets: list[set[str]]) -> DictVectorizer:
    """Fit a binary relation vocabulary on training encounters only."""
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    vectorizer.fit([{token: 1.0 for token in tokens} for tokens in token_sets])
    return vectorizer


def build_similarity_graph(
    frame: pd.DataFrame,
    vectorizer: DictVectorizer,
    neighbors: int = 8,
    similarity_threshold: float = 0.35,
) -> GraphProjection:
    """Build a k-nearest relation graph without O(N²) pair enumeration."""
    tokens = relation_token_sets(frame)
    vocabulary = set(vectorizer.vocabulary_)
    tokens = [values & vocabulary for values in tokens]
    node_count = len(frame)
    if node_count < 2:
        return GraphProjection(
            frame["encounter_id"].astype(str).to_numpy(),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            tokens,
            {"nodes": node_count, "undirected_edges": 0, "isolated_nodes": node_count},
        )
    # Build bounded candidate pools from an inverted relation-token index. For
    # very common tokens, only a deterministic local posting window is examined.
    # This avoids allocating or evaluating an all-pairs N x N similarity matrix.
    postings: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(tokens):
        for token in values:
            postings[token].append(index)
    posting_window = 120
    edge_weights: dict[tuple[int, int], float] = {}
    for source in range(node_count):
        candidate_indices: set[int] = set()
        for token in tokens[source]:
            token_postings = postings[token]
            if len(token_postings) <= posting_window * 2:
                candidate_indices.update(token_postings)
            else:
                position = bisect_left(token_postings, source)
                start = max(0, position - posting_window)
                stop = min(len(token_postings), position + posting_window + 1)
                candidate_indices.update(token_postings[start:stop])
        candidate_indices.discard(source)
        if not candidate_indices:
            candidate_indices.add((source + 1) % node_count)
        source_norm = np.sqrt(max(1, len(tokens[source])))
        candidates = []
        for target in candidate_indices:
            overlap = len(tokens[source] & tokens[target])
            similarity = overlap / (
                source_norm * np.sqrt(max(1, len(tokens[target])))
            )
            candidates.append((target, float(similarity)))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected = [item for item in candidates if item[1] >= similarity_threshold][
            :neighbors
        ]
        if not selected and candidates:
            selected = [candidates[0]]
        for target, weight in selected[:neighbors]:
            pair = (min(source, target), max(source, target))
            if pair[0] != pair[1]:
                edge_weights[pair] = max(edge_weights.get(pair, 0.0), weight)
    edges = np.asarray(sorted(edge_weights), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    weights = np.asarray(
        [edge_weights[tuple(edge)] for edge in edges], dtype=np.float32
    )
    degree = np.zeros(node_count, dtype=int)
    if len(edges):
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
    possible = node_count * (node_count - 1) / 2
    stats = {
        "nodes": node_count,
        "undirected_edges": int(len(edges)),
        "directed_edges_before_self_loops": int(2 * len(edges)),
        "density": float(len(edges) / possible) if possible else 0.0,
        "isolated_nodes": int((degree == 0).sum()),
        "mean_degree": float(degree.mean()),
        "maximum_degree": int(degree.max(initial=0)),
        "neighbors_requested": int(neighbors),
        "similarity_threshold": float(similarity_threshold),
        "relation_feature_count": int(len(vectorizer.feature_names_)),
        "candidate_search": "bounded inverted-index cosine kNN",
        "maximum_posting_window_per_token": int(posting_window * 2),
        "self_edges_before_normalization": 0,
    }
    return GraphProjection(
        frame["encounter_id"].astype(str).to_numpy(),
        edges,
        weights,
        tokens,
        stats,
    )


def normalized_adjacency(
    node_count: int, edges: np.ndarray, weights: np.ndarray
) -> sparse.coo_matrix:
    """Return D^-1/2 (A + I) D^-1/2 as a SciPy COO matrix."""
    if len(edges):
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        columns = np.concatenate([edges[:, 1], edges[:, 0]])
        values = np.concatenate([weights, weights]).astype(np.float32)
        adjacency = sparse.coo_matrix(
            (values, (rows, columns)), shape=(node_count, node_count), dtype=np.float32
        )
    else:
        adjacency = sparse.coo_matrix((node_count, node_count), dtype=np.float32)
    adjacency = adjacency + sparse.eye(node_count, dtype=np.float32, format="coo")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.power(degrees, -0.5, where=degrees > 0)
    inverse[~np.isfinite(inverse)] = 0.0
    diagonal = sparse.diags(inverse.astype(np.float32))
    return (diagonal @ adjacency @ diagonal).tocoo()


def neighbor_evidence(
    frame: pd.DataFrame, graph: GraphProjection
) -> dict[str, list[dict[str, Any]]]:
    """Create case-explorer neighbours with shared relation evidence."""
    neighbours: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (left, right), weight in zip(graph.edges, graph.weights, strict=True):
        neighbours[int(left)].append((int(right), float(weight)))
        neighbours[int(right)].append((int(left), float(weight)))
    encounters = frame["encounter_id"].astype(str).tolist()
    targets = frame["target"].astype(int).tolist()
    result: dict[str, list[dict[str, Any]]] = {}
    for index, encounter in enumerate(encounters):
        items = []
        for other, weight in sorted(
            neighbours.get(index, []), key=lambda item: item[1], reverse=True
        )[:8]:
            items.append(
                {
                    "encounter_id": encounters[other],
                    "similarity": weight,
                    "actual_target": targets[other],
                    "shared_relations": sorted(
                        graph.relation_tokens[index] & graph.relation_tokens[other]
                    ),
                }
            )
        result[encounter] = items
    return result
