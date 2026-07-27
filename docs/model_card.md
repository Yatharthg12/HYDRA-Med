# Model Card: HealthGraph Readmission Lab

## Intended use

This model set supports an academic comparison of tabular and graph-aware
methods for predicting readmission within 30 days after a diabetes-related
hospital encounter. It is intended for reproducibility study, software
demonstration, and methodological discussion.

It is not intended for diagnosis, treatment, triage, care denial, resource
allocation, or decisions about real patients.

## Data and population

The source is the UCI Diabetes 130-US Hospitals for Years 1999–2008 dataset.
The verified artifact set uses its deterministic 20,000-row sample. After mapped
death/expiration/hospice exclusions, 19,531 encounters from 17,690 patients
remain, with 11.39% positives overall.

The same patient-disjoint test population contains 2,889 encounters, 2,654
patients, and 330 positives.

## Models

- class-balanced Logistic Regression (`C=0.1`);
- randomized PCA selected from a validation-only sensitivity grid (50
  components, 57.0% achieved variance) followed by distance-weighted
  15-neighbour kNN;
- two-layer pure-PyTorch GCN over split-local weighted encounter graphs.

Each model's decision threshold maximizes validation positive-class F1.

## Verified performance

| Model | Threshold | PR-AUC | ROC-AUC | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5339 | 0.2447 | 0.6755 | 0.6265 | 0.2060 | 0.5030 | 0.2923 |
| PCA + kNN | 0.1265 | 0.1680 | 0.5913 | 0.5755 | 0.1519 | 0.5394 | 0.2370 |
| GCN | 0.4798 | 0.1866 | 0.6058 | 0.5688 | 0.1411 | 0.6394 | 0.2312 |

The GCN did not outperform Logistic Regression, although it exceeded PCA+kNN by
PR-AUC. Logistic Regression is the measured winner. Even its PR-AUC and
precision remain modest; the artifact set should not be described as strong
clinical performance.

Patient-clustered 95% bootstrap intervals for the primary ranking metrics are:

| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Recall (95% CI) | F1 (95% CI) |
|---|---|---|---|---|
| Logistic Regression | 0.2447 [0.2022, 0.2894] | 0.6755 [0.6409, 0.7074] | 0.5030 [0.4479, 0.5590] | 0.2923 [0.2554, 0.3276] |
| PCA + kNN | 0.1680 [0.1398, 0.2055] | 0.5913 [0.5556, 0.6252] | 0.5394 [0.4869, 0.5932] | 0.2370 [0.2077, 0.2650] |
| GCN | 0.1866 [0.1520, 0.2265] | 0.6058 [0.5735, 0.6392] | 0.6394 [0.5863, 0.6918] | 0.2312 [0.2052, 0.2575] |

Intervals use 1,000 paired resamples of complete test-patient clusters. They
quantify sampling uncertainty on this split and are not clinical confidence
guarantees.

## GCN relationship robustness

| Scenario | Trials | Mean edges | Mean PR-AUC | Mean ROC-AUC | Mean absolute probability shift |
|---|---:|---:|---:|---:|---:|
| Baseline | 10 | 16,524 | 0.1866 | 0.6058 | 0.0000 |
| Remove 5% | 10 | 15,698 | 0.1878 | 0.6070 | 0.0066 |
| Remove 10% | 10 | 14,872 | 0.1902 | 0.6083 | 0.0105 |
| Remove 20% | 10 | 13,219 | 0.1901 | 0.6072 | 0.0166 |
| Add 5% noise | 10 | 17,350 | 0.1866 | 0.6055 | 0.0088 |
| Add 10% noise | 10 | 18,176 | 0.1879 | 0.6074 | 0.0140 |

Small non-monotonic metric changes are expected across finite perturbations and
are not evidence that relationship corruption improves the model.
Probability shift increases as perturbation grows, while discrimination remains
within a narrow range. This limited stress test does not establish robustness
to real-world distribution shift.

## GCN seed stability and graph contribution

Across seeds 42, 52, 62, 72, and 82, mean GCN PR-AUC is 0.1800
(standard deviation 0.0107; 95% t interval [0.1667, 0.1933]). Mean recall is
0.6164 (standard deviation 0.0682; interval [0.5317, 0.7011]). Seed 42 remains
the primary comparison; no best-seed selection is performed.

The identity-only retrained GCN achieved PR-AUC 0.2370, compared with 0.1866 for
the original graph. A matched-random retrained graph achieved 0.1857.
Feature-shuffled inference fell to 0.1376. These findings do not support a claim
that the current clinical similarity neighbourhood improves prediction. The
graph-construction strategy requires refinement.

## Interpretability

The dashboard displays Logistic Regression coefficients, PCA explained
variance, GCN validation history, thresholds, curves, confusion matrices, and
case-level graph relation evidence. Coefficients and neighbours show model
structure, not causal clinical explanations.

## Important limitations

- Historical US data may not represent current practice or another institution.
- Race, gender, age, and care-use patterns may encode structural bias.
- There is no external, temporal, prospective, or clinical validation.
- The binary target combines `NO` and readmission after 30 days.
- Similarity edges are engineered from coded fields, not verified causal links.
- Grouped splitting prevents patient leakage but does not resolve site leakage;
  no hospital identifier is provided.
- Reduced mode sacrifices coverage for practical CPU demonstration.
- Intervals come from one deterministic split and do not represent external or
  temporal variability.
- The identity-only ablation outperformed the engineered graph.
- Five training seeds are too few to characterize all optimization variance.
- Robustness trials perturb edges synthetically and do not model missing
  clinical measurements or changing care practice.

## Governance

The dashboard labels every output as research-only. It contains no patient names
and fabricates no identities. Encounter and patient numbers remain de-identified
dataset references. Retraining is an explicit CLI action and never occurs during
page requests.
