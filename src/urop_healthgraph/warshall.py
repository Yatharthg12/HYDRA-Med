"""Warshall's transitive-closure demonstration on five healthcare entities."""

from __future__ import annotations

from typing import Any

import numpy as np


WARSHALL_NODES = ["Patient", "Disease", "Medication", "Lab Test", "Complication"]
INITIAL_ADJACENCY = np.asarray(
    [
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=bool,
)
EXPECTED_CLOSURE = np.asarray(
    [
        [0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
    ],
    dtype=bool,
)


def warshall_iterations(adjacency: np.ndarray | None = None) -> dict[str, Any]:
    """Calculate and verify T^(0) through T^(5), including logical evidence."""
    matrix = (
        INITIAL_ADJACENCY.copy()
        if adjacency is None
        else np.asarray(adjacency, dtype=bool)
    )
    if matrix.shape != (5, 5):
        raise ValueError("The healthcare demonstration requires a 5x5 matrix")
    iterations: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "step_index": 0,
            "notation": "T^(0)",
            "via": None,
            "intermediate_node": None,
            "matrix": matrix.astype(int).tolist(),
            "new_pairs": [],
            "new_cells": [],
            "description": (
                "T^(0) is the original adjacency matrix. It contains only the "
                "five direct relationships supplied to the demonstration."
            ),
        }
    ]
    for k, via in enumerate(WARSHALL_NODES):
        previous = matrix.copy()
        for i in range(len(WARSHALL_NODES)):
            for j in range(len(WARSHALL_NODES)):
                matrix[i, j] = matrix[i, j] or (matrix[i, k] and matrix[k, j])
        added = np.argwhere(matrix & ~previous)
        pairs = []
        for i, j in added:
            source = WARSHALL_NODES[int(i)]
            target = WARSHALL_NODES[int(j)]
            pairs.append(
                {
                    "row": int(i),
                    "column": int(j),
                    "from": source,
                    "to": target,
                    "via": via,
                    "plain_english": (
                        f"{source} can reach {target} through {via}."
                    ),
                    "calculation": {
                        "previous_value": int(previous[i, j]),
                        "source_to_intermediate": int(previous[i, k]),
                        "intermediate_to_target": int(previous[k, j]),
                        "result": int(matrix[i, j]),
                        "expression": (
                            f"T_previous[{source}, {target}] OR "
                            f"(T_previous[{source}, {via}] AND "
                            f"T_previous[{via}, {target}])"
                        ),
                    },
                }
            )
        description = (
            f"Using {via} as the intermediate reveals "
            + (
                ", ".join(f"{item['from']} -> {item['to']}" for item in pairs)
                if pairs
                else "no new reachable pairs"
            )
            + "."
        )
        iterations.append(
            {
                "iteration": k + 1,
                "step_index": k + 1,
                "notation": f"T^({k + 1})",
                "via": via,
                "intermediate_node": via,
                "matrix": matrix.astype(int).tolist(),
                "new_pairs": pairs,
                "new_cells": [
                    {"row": item["row"], "column": item["column"]}
                    for item in pairs
                ],
                "description": description,
            }
        )
    if adjacency is None and not np.array_equal(matrix, EXPECTED_CLOSURE):
        raise AssertionError("Warshall closure does not match the verified result")
    return {
        "nodes": WARSHALL_NODES,
        "direct_edges": [
            {
                "from": WARSHALL_NODES[int(i)],
                "to": WARSHALL_NODES[int(j)],
            }
            for i, j in np.argwhere(INITIAL_ADJACENCY)
        ],
        "iterations": iterations,
        "verified": bool(
            adjacency is not None or np.array_equal(matrix, EXPECTED_CLOSURE)
        ),
        "time_complexity": "O(V^3)",
        "memory_complexity": "O(V^2)",
        "scope_note": (
            "This five-node example demonstrates transitive reachability only. "
            "It is not applied to the full hospital graph and does not produce "
            "readmission predictions."
        ),
    }
