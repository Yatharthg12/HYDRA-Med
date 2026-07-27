# Statistical Analysis

## Purpose

This analysis quantifies uncertainty in the three held-out model results while
respecting repeated encounters from the same patient. It also compares models
with paired resampling so a replicate never gives one model a different test
population from another.

## Patient-clustered paired bootstrap

The test population contains 2,889 encounters from 2,654 patients. For each of
1,000 replicates:

1. sample 2,654 test-patient identifiers with replacement;
2. include every test encounter belonging to each sampled patient;
3. repeat a patient's complete encounter cluster when that patient is sampled
   more than once;
4. use the exact same encounter index vector for all three models;
5. apply each model's validation-selected threshold; and
6. skip and count a replicate if its sampled labels contain only one class.

The master seed is 42042. All 1,000 replicates were valid in the verified run;
zero were skipped. The SHA-256 digest of the generated index sequence is stored
in the JSON artifact.

Naive encounter-level bootstrap is not used because it would treat correlated
encounters from one patient as independent. The clustered design addresses this
specific dependence but does not account for hospital-level clustering because
the source has no hospital identifier.

## Confidence intervals

The bounds are the 2.5th and 97.5th percentiles of valid bootstrap estimates.

| Model | Metric | Estimate | 95% percentile interval |
|---|---|---:|---|
| Logistic Regression | PR-AUC | 0.2447 | [0.2022, 0.2894] |
| Logistic Regression | ROC-AUC | 0.6755 | [0.6409, 0.7074] |
| Logistic Regression | Accuracy | 0.7217 | [0.7054, 0.7380] |
| Logistic Regression | Balanced accuracy | 0.6265 | [0.5975, 0.6552] |
| Logistic Regression | Precision | 0.2060 | [0.1770, 0.2352] |
| Logistic Regression | Recall | 0.5030 | [0.4479, 0.5590] |
| Logistic Regression | F1 | 0.2923 | [0.2554, 0.3276] |
| PCA + kNN | PR-AUC | 0.1680 | [0.1398, 0.2055] |
| PCA + kNN | ROC-AUC | 0.5913 | [0.5556, 0.6252] |
| PCA + kNN | Accuracy | 0.6033 | [0.5859, 0.6197] |
| PCA + kNN | Balanced accuracy | 0.5755 | [0.5469, 0.6054] |
| PCA + kNN | Precision | 0.1519 | [0.1314, 0.1728] |
| PCA + kNN | Recall | 0.5394 | [0.4869, 0.5932] |
| PCA + kNN | F1 | 0.2370 | [0.2077, 0.2650] |
| GCN | PR-AUC | 0.1866 | [0.1520, 0.2265] |
| GCN | ROC-AUC | 0.6058 | [0.5735, 0.6392] |
| GCN | Accuracy | 0.5144 | [0.4972, 0.5317] |
| GCN | Balanced accuracy | 0.5688 | [0.5423, 0.5974] |
| GCN | Precision | 0.1411 | [0.1235, 0.1595] |
| GCN | Recall | 0.6394 | [0.5863, 0.6918] |
| GCN | F1 | 0.2312 | [0.2052, 0.2575] |

## Paired differences

Differences are calculated replicate by replicate as left model minus right
model. `P(diff > 0)` is the proportion of valid bootstrap differences above
zero. The two-sided tail probability doubles the smaller smoothed tail
proportion and is reported as a descriptive bootstrap probability. It must not
be presented as proof of clinical significance.

| Comparison | Metric | Observed | Bootstrap mean | 95% interval | P(diff > 0) | Two-sided tail |
|---|---|---:|---:|---|---:|---:|
| Logistic − GCN | PR-AUC | +0.0582 | +0.0569 | [0.0245, 0.0926] | 1.000 | 0.002 |
| Logistic − GCN | ROC-AUC | +0.0696 | +0.0686 | [0.0335, 0.1013] | 1.000 | 0.002 |
| Logistic − GCN | Recall | −0.1364 | −0.1378 | [−0.1982, −0.0776] | 0.000 | 0.002 |
| Logistic − GCN | F1 | +0.0610 | +0.0603 | [0.0316, 0.0883] | 1.000 | 0.002 |
| Logistic − PCA+kNN | PR-AUC | +0.0767 | +0.0744 | [0.0415, 0.1074] | 1.000 | 0.002 |
| Logistic − PCA+kNN | ROC-AUC | +0.0841 | +0.0833 | [0.0465, 0.1175] | 1.000 | 0.002 |
| Logistic − PCA+kNN | Recall | −0.0364 | −0.0372 | [−0.0981, 0.0249] | 0.106 | 0.262 |
| Logistic − PCA+kNN | F1 | +0.0552 | +0.0548 | [0.0256, 0.0842] | 1.000 | 0.002 |
| GCN − PCA+kNN | PR-AUC | +0.0185 | +0.0175 | [−0.0166, 0.0485] | 0.860 | 0.282 |
| GCN − PCA+kNN | ROC-AUC | +0.0145 | +0.0148 | [−0.0248, 0.0557] | 0.771 | 0.460 |
| GCN − PCA+kNN | Recall | +0.1000 | +0.1006 | [0.0344, 0.1724] | 0.999 | 0.004 |
| GCN − PCA+kNN | F1 | −0.0058 | −0.0055 | [−0.0323, 0.0216] | 0.337 | 0.675 |

Logistic Regression has higher PR-AUC, ROC-AUC, and F1 than both alternatives
under the paired resamples. GCN has higher recall than both alternatives. The
GCN–PCA+kNN intervals for PR-AUC, ROC-AUC, and F1 include zero.

## Artifacts and tests

Machine-readable results:

- `artifacts/metrics/bootstrap_confidence_intervals.json`
- `artifacts/metrics/bootstrap_confidence_intervals.csv`
- `artifacts/metrics/paired_model_differences.json`
- `artifacts/metrics/paired_model_differences.csv`

Tests verify complete cluster resampling, common paired populations, stable
schemas, invalid single-class handling, deterministic hashes, zero differences
for identical predictions, and ordered interval bounds.
