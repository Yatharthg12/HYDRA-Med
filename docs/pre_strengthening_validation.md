# Pre-Strengthening Validation

This document records the verified repository state before the statistical,
robustness, ablation, PCA-sensitivity, and Warshall-dashboard strengthening
work began.

## Execution context

- Validation timestamp: 2026-07-27T18:18:51+05:30
- Operating system: Windows
- Python: 3.13.2
- Dataset mode: reduced
- Cached experiment command: `python run_experiments.py --dataset reduced`
- Cached command wall time: 22.172 seconds
- Original forced artifact run time recorded by the manifest: 69.301 seconds

The experiment command successfully verified and reused the complete reduced
artifact set. It did not retrain because the manifest already identified a
complete reduced-mode run.

## Dependency versions

| Package | Version |
|---|---:|
| pandas | 2.3.2 |
| NumPy | 2.2.6 |
| SciPy | 1.16.2 |
| scikit-learn | 1.7.2 |
| NetworkX | 3.6.1 |
| Flask | 3.1.2 |
| joblib | 1.5.2 |
| Matplotlib | 3.10.6 |
| PyTorch | 2.11.0 |
| pytest | 9.0.3 |

## Automated tests

Command:

```powershell
python -m pytest -q
```

Result:

```text
11 passed in 7.47s
```

## Cohort and split verification

- Source reduced sample: 20,000 encounters
- Eligible encounters: 19,531
- Eligible patients: 17,690
- Positive prevalence: 11.392%
- End-of-life/hospice exclusions: 469
- Independently recomputed cross-split patient overlap: 0

| Split | Encounters | Patients | Positive encounters | Prevalence |
|---|---:|---:|---:|---:|
| Train | 13,715 | 12,383 | 1,559 | 11.367% |
| Validation | 2,927 | 2,653 | 336 | 11.479% |
| Test | 2,889 | 2,654 | 330 | 11.423% |

All three saved pairwise patient-overlap counts were zero.

## Primary model metrics

These values were read from
`artifacts/metrics/model_comparison.json`.

| Model | PR-AUC | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.244746 | 0.675452 | 0.721703 | 0.626466 | 0.205955 | 0.503030 | 0.292254 | 0.533889 |
| Graph Convolutional Network | 0.186566 | 0.605815 | 0.514365 | 0.568818 | 0.141137 | 0.639394 | 0.231233 | 0.479847 |
| PCA + k-Nearest Neighbours | 0.168035 | 0.591344 | 0.603323 | 0.575480 | 0.151877 | 0.539394 | 0.237017 | 0.126494 |

Logistic Regression was the overall measured winner by PR-AUC and ROC-AUC.
The GCN had the highest positive-class recall. No interpretation was changed to
favour the graph model.

## Graph statistics

- Heterogeneous graph nodes: 37,278
- Heterogeneous graph edges: 131,054
- Test encounter-projection nodes: 2,889
- Test undirected projection edges: 16,524
- Test isolated nodes: 0
- Cross-split projection edges: 0 by construction

## Live dashboard validation

The application was started with:

```powershell
python app.py
```

The following live HTTP requests returned status 200:

| Route or endpoint | Status |
|---|---:|
| `/` | 200 |
| `/dataset` | 200 |
| `/warshall` | 200 |
| `/models` | 200 |
| `/graph` | 200 |
| `/robustness` | 200 |
| `/cases` | 200 |
| `/limitations` | 200 |
| `/api/dataset` | 200 |
| `/api/models` | 200 |
| `/api/warshall` | 200 |
| `/api/graph` | 200 |
| `/api/robustness` | 200 |
| `/api/cases?q=1` | 200 |
| `/api/health` | 200 |

## Pre-change conclusion

The core reduced pipeline, saved split, three primary models, graph artifacts,
automated tests, and Flask routes were operational before strengthening.
Remaining work concerned statistical uncertainty, repeated analyses, graph and
PCA ablations, richer research artifacts, and the reported Warshall UI
synchronization defect.
