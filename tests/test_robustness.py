import numpy as np

from urop_healthgraph.gcn_model import TwoLayerGCN
from urop_healthgraph.graph_builder import GraphProjection
from urop_healthgraph.robustness import run_repeated_robustness


def _small_projection() -> GraphProjection:
    return GraphProjection(
        node_ids=np.asarray([str(index) for index in range(8)]),
        edges=np.asarray(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
            dtype=np.int64,
        ),
        weights=np.ones(8, dtype=np.float32),
        relation_tokens=[set() for _ in range(8)],
        statistics={},
    )


def test_repeated_robustness_is_reproducible_and_schema_is_consistent() -> None:
    model = TwoLayerGCN(input_features=3, hidden_size=4, dropout=0.0)
    features = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    arguments = (
        model,
        features,
        labels,
        _small_projection(),
        0.5,
        (0.1, 0.2),
        (0.1,),
        (7, 9, 11),
    )
    trials_a, summary_a = run_repeated_robustness(*arguments)
    trials_b, summary_b = run_repeated_robustness(*arguments)
    assert trials_a == trials_b
    assert summary_a == summary_b
    assert len(trials_a) == 12
    assert all(row["number_of_trials"] == 3 for row in summary_a)
    assert all(row["metrics"]["pr_auc"]["ci_95_lower"] <= row["metrics"]["pr_auc"]["ci_95_upper"] for row in summary_a)
    assert all(row["probability_shift_mean_absolute"] == 0 for row in trials_a if row["scenario"] == "baseline")
