# Repeated Graph Robustness Protocol

## Question

The experiment asks how the fixed primary GCN probabilities change when only
the test encounter graph is corrupted. It does not test missing clinical
features, altered outcomes, changing patients, retraining, or clinical
distribution shift.

## Fixed elements

For every trial, the following are unchanged:

- primary GCN trained with seed 42;
- node feature matrix;
- node labels;
- node count and test-node identities;
- validation-selected threshold; and
- baseline probabilities used for probability-shift calculation.

## Scenarios and seeds

Six scenarios are evaluated:

- baseline graph;
- remove 5% of undirected edges;
- remove 10%;
- remove 20%;
- add 5% simple random noise edges; and
- add 10% simple random noise edges.

Each scenario records trial seeds 42 through 51. A scenario-specific derived
seed avoids reusing the same random stream across perturbation types. The
baseline is intentionally identical across its ten records and acts as the
common reference.

Removal samples existing edge positions without replacement. Noise addition
creates canonical undirected pairs, excludes self-edges and existing edges, and
prevents duplicates. Added edges receive the median baseline edge weight. Every
trial recomputes `D^(-1/2)(A + I)D^(-1/2)`.

## Measurements

Each trial records seed, derived seed, node count, edges before and after,
PR-AUC, ROC-AUC, balanced accuracy, recall, F1, and mean absolute probability
shift from baseline. Each scenario reports count, mean, sample standard
deviation, median, minimum, maximum, and a two-sided 95% Student-t interval.

## Verified results

| Scenario | n | Mean PR-AUC ± SD | PR-AUC 95% interval | Mean recall | Mean F1 | Probability shift 95% interval |
|---|---:|---:|---|---:|---:|---|
| Baseline | 10 | 0.1866 ± 0.0000 | [0.1866, 0.1866] | 0.6394 | 0.2312 | [0.0000, 0.0000] |
| Remove 5% | 10 | 0.1878 ± 0.0023 | [0.1861, 0.1894] | 0.6385 | 0.2323 | [0.0064, 0.0067] |
| Remove 10% | 10 | 0.1902 ± 0.0024 | [0.1884, 0.1919] | 0.6364 | 0.2322 | [0.0102, 0.0107] |
| Remove 20% | 10 | 0.1901 ± 0.0025 | [0.1883, 0.1919] | 0.6294 | 0.2318 | [0.0162, 0.0169] |
| Add 5% noise | 10 | 0.1866 ± 0.0027 | [0.1847, 0.1885] | 0.6348 | 0.2320 | [0.0085, 0.0090] |
| Add 10% noise | 10 | 0.1879 ± 0.0032 | [0.1857, 0.1902] | 0.6367 | 0.2354 | [0.0136, 0.0144] |

Mean probability shift grows with perturbation magnitude. Discrimination and
threshold metrics fluctuate non-monotonically. Those small increases must not
be interpreted as evidence that removing or adding erroneous relationships
improves the model; they can arise from finite perturbation variance and the
weak contribution of the current graph.

## Artifacts

- raw: `artifacts/metrics/robustness_trials.{json,csv}`
- aggregate: `artifacts/metrics/robustness_summary.{json,csv}`
- compatibility summary: `artifacts/metrics/robustness_results.{json,csv}`
- figure: `artifacts/figures/robustness.png`
