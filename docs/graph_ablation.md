# Graph Contribution Ablation

## Design

The ablation study examines whether the GCN benefits from the engineered
clinical similarity neighbourhood rather than only from its node features and
architecture. All conditions use the same saved split, test nodes, test labels,
feature schema, hidden width, optimizer settings, and validation-only selection
protocol.

## Conditions

### Original clinical similarity graph

The prespecified seed-42 GCN aggregates over split-local edges constructed from
diagnosis categories, active medications, admission type, and admission source.
This is the primary comparison model.

### Identity-only adjacency

The graph contains normalization self-loops only. Nodes cannot receive neighbour
messages. Because the training operator changes, the GCN is retrained on the
identity-only training adjacency, checked on identity-only validation adjacency,
and evaluated once on identity-only test adjacency. Epoch and threshold are
chosen without test data.

### Matched random graph

Each split receives a deterministic simple undirected random graph with the same
node and edge counts as its original projection. Self-edges and duplicate edges
are prohibited; weights are resampled from that split's original weights. A new
GCN is trained and selected using the corresponding random train and validation
graphs.

### Feature-shuffled inference

The primary GCN and original test graph remain fixed. Test feature rows are
permuted with a deterministic seed while graph edges and labels stay in place.
This inference-only condition disrupts node-feature alignment and performs no
retraining or test-set selection.

## Results

| Condition | Epoch | Threshold | Test edges | PR-AUC | ROC-AUC | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original graph | 40 | 0.4798 | 16,524 | 0.1866 | 0.6058 | 0.5688 | 0.1411 | 0.6394 | 0.2312 |
| Identity-only | 23 | 0.5223 | 0 | 0.2370 | 0.6830 | 0.6349 | 0.1925 | 0.5879 | 0.2900 |
| Matched random | 68 | 0.4669 | 16,524 | 0.1857 | 0.6200 | 0.5801 | 0.1491 | 0.6061 | 0.2394 |
| Feature shuffled | 40 | 0.4798 | 16,524 | 0.1376 | 0.5319 | 0.5314 | 0.1299 | 0.4606 | 0.2027 |

## Answers to the ablation questions

**Does neighbourhood information improve the GCN compared with no neighbour
information?** No in this experiment. Identity-only adjacency improves PR-AUC
by 0.0504 and F1 by 0.0588 relative to the original graph.

**Does the clinically constructed graph perform differently from a random
graph?** Its PR-AUC is nearly identical: 0.1866 versus 0.1857. The random graph
has higher ROC-AUC, balanced accuracy, and F1 in this run. This is not evidence
that random graphs are generally preferable; it is evidence that this study has
not established benefit from the constructed edge semantics.

**How dependent is performance on feature–graph alignment?** Shuffling features
reduces PR-AUC by 0.0490 and ROC-AUC by 0.0739. The trained model depends on
feature alignment, but that fact does not demonstrate useful neighbour
contribution because identity-only still performs best.

## Limitations

The study uses one ablation seed, one deterministic split, engineered
similarities, and a compact GCN. Identity and random conditions are retrained,
so differences reflect the combined effect of graph structure and the resulting
optimization path. Broader graph definitions, edge-feature models, inductive
evaluation, additional ablation seeds, and external datasets are needed.

Results are saved in `artifacts/metrics/graph_ablation_results.{json,csv}`.
