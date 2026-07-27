# Final Results Summary

## Verified run

The final forced research command was:

```powershell
python run_experiments.py --dataset reduced --research --force
```

It completed successfully in **206.44 seconds** on Windows with Python 3.13.2.
The reduced source sample contains 20,000 encounters. Mapped
death/expiration/hospice eligibility rules exclude 469, leaving 19,531
encounters from 17,690 patients with 11.39% positive prevalence.

The grouped split has 13,715/2,927/2,889 train/validation/test encounters and
12,383/2,653/2,654 patients. Pairwise patient overlap is zero.

## Primary single-seed model comparison

| Rank | Model | PR-AUC | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.2447 | 0.6755 | 0.7217 | 0.6265 | 0.2060 | 0.5030 | 0.2923 |
| 2 | GCN, seed 42 | 0.1866 | 0.6058 | 0.5144 | 0.5688 | 0.1411 | 0.6394 | 0.2312 |
| 3 | PCA + kNN | 0.1680 | 0.5913 | 0.6033 | 0.5755 | 0.1519 | 0.5394 | 0.2370 |

Logistic Regression is the measured overall winner. GCN has the highest
positive-class recall. The results do not support claiming that GCN is the
superior predictive model.

## Bootstrap confidence intervals

All 1,000 patient-clustered paired replicates were valid. PR-AUC intervals are
Logistic Regression [0.2022, 0.2894], GCN [0.1520, 0.2265], and PCA+kNN
[0.1398, 0.2055]. ROC-AUC intervals are [0.6409, 0.7074], [0.5735, 0.6392],
and [0.5556, 0.6252], respectively.

The paired Logistic-minus-GCN difference is +0.0582 for PR-AUC
([0.0245, 0.0926]) and −0.1364 for recall
([−0.1982, −0.0776]). The paired GCN-minus-PCA+kNN PR-AUC interval includes
zero ([−0.0166, 0.0485]), while its recall difference is +0.1000
([0.0344, 0.1724]). Bootstrap tail probabilities are descriptive, not clinical
significance tests.

## GCN multi-seed stability

Across seeds 42, 52, 62, 72, and 82:

| Metric | Mean ± SD | Range | 95% t interval |
|---|---|---|---|
| PR-AUC | 0.1800 ± 0.0107 | 0.1616–0.1874 | [0.1667, 0.1933] |
| ROC-AUC | 0.6019 ± 0.0094 | 0.5857–0.6089 | [0.5902, 0.6136] |
| Balanced accuracy | 0.5664 ± 0.0052 | 0.5572–0.5699 | [0.5599, 0.5729] |
| Recall | 0.6164 ± 0.0682 | 0.5242–0.6818 | [0.5317, 0.7011] |
| F1 | 0.2295 ± 0.0037 | 0.2229–0.2318 | [0.2249, 0.2341] |

Seed 42 remains primary; the maximum seed result is not substituted.

## Repeated graph robustness

Ten trials per scenario show increasing probability shifts with stronger edge
perturbation. Mean shifts are 0.0066/0.0105/0.0166 for 5%/10%/20% removal and
0.0088/0.0140 for 5%/10% noise. Mean PR-AUC ranges from 0.1866 to 0.1902.
Non-monotonic fluctuations are treated as perturbation variance, not benefit
from corruption.

## PCA sensitivity

Component counts 25, 50, 75, and 100 retain 36.35%, 57.00%, 76.01%, and 92.84%
variance. Validation PR-AUC values are 0.1637, 0.1828, 0.1823, and 0.1750.
The validation-only rule selects 50 components with 15 neighbours. The
150-component request is skipped because only 137 are allowed.

## Graph ablation

Identity-only adjacency achieves PR-AUC 0.2370, above the original graph's
0.1866. The matched random graph reaches 0.1857. Feature-shuffled inference
falls to 0.1376. The current graph does not show predictive benefit over no
neighbour aggregation and requires refinement.

## Software verification

The final automated suite reports **19 passed**. A live Flask process returned
HTTP 200 for all eight dashboard pages; dataset, model, statistics, Warshall,
graph, robustness, PCA, ablation, case-search, and health APIs; and
representative CSV and JSON download URLs.

The Warshall static integration contract confirms step-zero initialization,
both slider events, synchronized controls, and the expected DOM identifiers.

## Interpretation limits

The dataset is historical, site identifiers are unavailable, reduced mode is a
deterministic subset, no external or prospective validation exists, and all
models have modest positive precision. Bootstrap intervals quantify patient
sampling on one split, not deployment, temporal, institutional, or clinical
uncertainty. No result should guide care for an individual.
