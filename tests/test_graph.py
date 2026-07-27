import numpy as np
import torch

from urop_healthgraph.gcn_model import TwoLayerGCN, scipy_to_torch_sparse
from urop_healthgraph.graph_builder import (
    build_similarity_graph,
    fit_relation_schema,
    normalized_adjacency,
    relation_token_sets,
)
from urop_healthgraph.graph_ablation import random_simple_graph


def test_similarity_graph_has_valid_edges(clinical_frame) -> None:
    schema = fit_relation_schema(relation_token_sets(clinical_frame.iloc[:12]))
    graph = build_similarity_graph(
        clinical_frame, schema, neighbors=3, similarity_threshold=0.2
    )
    assert graph.statistics["nodes"] == len(clinical_frame)
    assert len(graph.edges) > 0
    assert np.all(graph.edges[:, 0] != graph.edges[:, 1])
    assert graph.statistics["self_edges_before_normalization"] == 0


def test_normalized_adjacency_and_gcn_forward_shape(clinical_frame) -> None:
    schema = fit_relation_schema(relation_token_sets(clinical_frame))
    graph = build_similarity_graph(clinical_frame, schema, neighbors=2)
    adjacency = normalized_adjacency(
        len(clinical_frame), graph.edges, graph.weights
    )
    assert adjacency.shape == (len(clinical_frame), len(clinical_frame))
    torch_adjacency = scipy_to_torch_sparse(adjacency)
    model = TwoLayerGCN(input_features=6, hidden_size=4, dropout=0.0)
    output = model(torch.randn(len(clinical_frame), 6), torch_adjacency)
    assert output.shape == (len(clinical_frame),)


def test_matched_random_graph_is_simple_exact_and_reproducible() -> None:
    first = random_simple_graph(30, 70, np.asarray([0.3, 0.7]), 44)
    second = random_simple_graph(30, 70, np.asarray([0.3, 0.7]), 44)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 70
    assert np.all(first[0][:, 0] < first[0][:, 1])
    assert len({tuple(edge) for edge in first[0].tolist()}) == 70
