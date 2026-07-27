# Methodology

## Research question and outcome

The implemented question is whether relationships among diabetes-related
hospital encounters add useful information when predicting readmission within
30 days. An encounter is positive only when the source value `readmitted` is
`<30`; `NO` and `>30` are negative. Because medication, treatment, and discharge
fields are used, the correct framing is a **discharge-time risk model**.

The project does not predict diabetes onset and makes no causal or clinical
readiness claim.

## Data audit and eligibility

The raw `?` marker is converted to missing data. `IDS_mapping.csv` is parsed as
three named sections, producing readable descriptions for admission type,
discharge disposition, and admission source.

Readmission is not a meaningful follow-up outcome for encounters ending in death
or hospice care. Eligibility is therefore determined by searching mapped
discharge descriptions for death, expiration, or hospice language. No hidden
numeric-ID list is used. In the verified reduced run, this excluded 469 of
20,000 sampled encounters.

Only `weight` is removed for missingness (over 96% missing). `payer_code` and
`medical_specialty` remain and receive explicit `Unknown` values. Missing A1C
and maximum glucose results become `Not Measured`, distinguishing an absent test
from an unknown result.

## Feature engineering

- `encounter_id` and `patient_nbr` remain metadata and never enter a predictor
  matrix.
- Age intervals become numeric midpoints while the readable age group remains
  available for the dashboard.
- ICD-9 primary, secondary, and third diagnosis values map to Diabetes,
  Circulatory, Respiratory, Digestive, Genitourinary, Musculoskeletal, Injury,
  Neoplasm, Other, or Unknown.
- Every medication status becomes a compact active indicator: `Up`, `Down`, and
  `Steady` are active; `No` is inactive.
- Numeric missing values use medians learned on training data.
- Categorical missing values use `Unknown` except the explicit lab category.
- Categories below 0.5% training frequency become `Other`.
- Numeric predictors are standardized; categorical predictors are one-hot
  encoded with unseen-category handling.

All fitting occurs on the training population. The fitted transforms are
serialized inside the tabular pipelines and separately for the GCN.

## Patient-safe splitting

Patients, not rows, are divided with seed 42. A patient-level positive stratum
indicates whether any eligible encounter is positive. Approximately 70% of
patients go to training and the remainder is divided evenly between validation
and test. All encounters inherit their patient's assignment.

Verified reduced split:

| Split | Encounters | Patients | Positives |
|---|---:|---:|---:|
| Train | 13,715 | 12,383 | 1,559 |
| Validation | 2,927 | 2,653 | 336 |
| Test | 2,889 | 2,654 | 330 |

All three pairwise patient-overlap counts are zero. The assignment is saved to
`data/processed/split_assignments.csv` and reused everywhere.

## Predictive models

### Logistic Regression

The baseline uses class-balanced Logistic Regression after shared feature
engineering. `C ∈ {0.1, 1.0}` is chosen by validation PR-AUC. The validation
threshold maximizing positive-class F1 is saved. Test results are recorded at
both that threshold and 0.5. Readable strongest positive and negative
coefficients are extracted; they are associations, not causal effects.

### PCA + k-Nearest Neighbours

A dense version of the training-fitted engineered matrix, including one-hot
columns, is standardized, then randomized PCA is fitted with seed 42. Component
counts 25, 50, 75, and 100 are evaluated; 150 is skipped because the training
matrix permits at most 137. For each count, distance-weighted kNN chooses among
7, 15, and 31 neighbours using validation PR-AUC. The final count maximizes
validation PR-AUC, with validation ROC-AUC and then fewer components as
tie-breakers. Test results are evaluated only after selection.

The verified run selected 50 components and 15 neighbours. The selected
configuration retains 57.00% variance. The 100-component configuration retains
92.84% but has lower validation PR-AUC, so variance retention is not treated as
the selection objective.

### Graph Convolutional Network

Each encounter is a node with the same engineered clinical vector and binary
label. Diagnosis categories, active medications, admission type, and admission
source form relation tokens. The token schema is fitted on training encounters.

For each split independently, a bounded inverted index identifies cosine
similarity candidates. At most eight neighbours above similarity 0.35 are kept;
the strongest available candidate prevents isolation where possible. Edges are
symmetrized and weighted by cosine similarity. Self-edges are absent until
normalization. No graph edge crosses populations.

The pure-PyTorch GCN applies:

1. graph convolution;
2. ReLU;
3. dropout;
4. second graph convolution producing one logit.

The adjacency is `D^(-1/2)(A + I)D^(-1/2)`. Weighted
`BCEWithLogitsLoss`, Adam, weight decay, deterministic seeds, and early stopping
on validation PR-AUC are used. The saved model is a state dictionary plus an
explicit architecture configuration.

## Heterogeneous representation

The research graph includes Patient, Encounter, DiagnosisCategory, Medication,
AdmissionType, AdmissionSource, and observed LabResultCategory nodes. Relations
are `has_encounter`, `diagnosed_with`, `received_medication`, `admitted_as`,
`originated_from`, and `has_lab_result`.

The complete reduced representation has 37,278 nodes and 131,054 edges. It is
counted with sparse/tabular structures. NetworkX materializes only the sampled
browser graph. Doctor nodes are not created because no individual doctor
identifier exists.

## Robustness

With clinical vectors, labels, weights, and trained GCN parameters fixed, test
edges are removed at 5%, 10%, and 20%, or random noise edges are added at 5% and
10%. Every scenario uses ten fixed perturbation seeds. Duplicate edges and
unintended self-edges are rejected, and normalized adjacency is recomputed for
each trial. The same primary trained GCN is used throughout.

Trial-level and aggregate results include PR-AUC, ROC-AUC, recall, F1, balanced
accuracy, edge count, and mean absolute probability shift from the baseline
probabilities. Summaries report mean, standard deviation, median, minimum,
maximum, and a 95% t interval. Small random improvements under corruption are
treated as perturbation variance, not evidence that corrupted graphs are
better. This is a relationship-quality stress test, not a missing-clinical-data
or distribution-shift experiment.

## Statistical reliability

The final test uncertainty analysis uses 1,000 patient-clustered paired
bootstrap replicates with master seed 42042. Unique test patients are sampled
with replacement; all encounters belonging to each selected patient are
included. If a patient is sampled repeatedly, the complete encounter cluster is
repeated accordingly.

The identical encounter index vector is used for Logistic Regression, PCA+kNN,
and GCN in each replicate. This pairing preserves within-replicate comparisons.
Replicates containing only one target class are skipped and counted. Percentile
95% intervals are calculated for PR-AUC, ROC-AUC, accuracy, balanced accuracy,
positive precision, positive recall, and positive F1. Paired differences are
reported for all three model pairs. The two-sided bootstrap tail probability is
labelled descriptive and is not interpreted as clinical proof.

## GCN training stability

The train/validation/test split, preprocessing, relation schema, graph
configuration, and training protocol remain fixed while the GCN is fitted with
seeds 42, 52, 62, 72, and 82. Seed 42 is the prespecified primary result.
The other runs quantify training variability; the best seed is never substituted
into the primary model comparison. The study records histories, selected epochs,
thresholds, durations, confusion matrices, per-seed metrics, and descriptive
statistics with 95% t intervals.

## Graph-contribution ablation

Four test conditions use identical test nodes and labels:

1. the primary GCN on the original clinical similarity graph;
2. a GCN retrained on identity-only adjacency, preventing neighbour aggregation;
3. a GCN retrained on a simple random graph with the original node and edge
   counts and resampled edge weights; and
4. the primary GCN with test feature rows deterministically shuffled while
   original edges and labels remain fixed.

Identity and random conditions require retraining because their message-passing
operator changes throughout learning. Their epochs and thresholds are selected
on matching validation graphs. Feature shuffling is an inference-only
feature–graph alignment stress test. No condition is retrained on test data.

The identity-only condition outperformed the original graph (PR-AUC 0.2370
versus 0.1866), and the matched random graph was nearly tied with the original
(0.1857). Current results therefore do not demonstrate useful predictive
contribution from the engineered neighbour structure. The feature-shuffled
decline to 0.1376 indicates sensitivity to feature alignment.

## Evaluation and interpretation

Every model reports ROC-AUC, PR-AUC, accuracy, balanced accuracy, positive
precision, positive recall, F1, a confusion matrix, threshold, and runtimes.
ROC and precision-recall coordinates are saved. PR-AUC is the primary ranking
metric because only 11.42% of test encounters are positive.

The measured ranking is Logistic Regression, GCN, then PCA+kNN by PR-AUC. This
outcome is retained rather than forcing the hypothesized graph method to win.
