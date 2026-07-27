# A Hybrid Framework of Graph Algorithms and AI for Relational Modeling of Healthcare Data

> **Project application:** HealthGraph Readmission Lab  
> **Research area:** Healthcare analytics, graph algorithms, machine learning, and graph neural networks  
> **Primary task:** Predict 30-day hospital readmission after a diabetes-related encounter  
> **Status:** Reproducible academic research prototype  
> **Use classification:** Educational viewing and authorized evaluation only

---

## Abstract

Hospital readmission within 30 days is an important indicator of care
continuity, patient risk, and healthcare resource utilization. Conventional
machine-learning systems usually represent hospital encounters as independent
rows in a table. That representation is useful, but it cannot directly describe
relationships created by shared diagnosis categories, medication patterns,
admission characteristics, and patient histories.

This project develops a hybrid framework that combines conventional predictive
modelling, graph-based healthcare representation, a classical graph algorithm,
and graph neural learning. Hospital encounters from the UCI Diabetes 130-US
Hospitals for Years 1999–2008 dataset are processed through a leakage-aware,
patient-grouped experimental pipeline. The system compares:

1. class-balanced Logistic Regression;
2. standardized Principal Component Analysis followed by distance-weighted
   k-Nearest Neighbours; and
3. a genuine two-layer Graph Convolutional Network operating on split-local
   encounter similarity graphs.

The project also constructs an explicit heterogeneous healthcare graph, presents
Warshall's transitive-closure algorithm as a small mathematical demonstration,
and measures GCN robustness under missing and noisy graph relationships. Its
statistical layer adds patient-clustered paired bootstrap intervals, repeated
GCN training across prespecified seeds, repeated graph perturbations, graph
ablations, and validation-only PCA component sensitivity. All results are saved
as reproducible artifacts and presented through a Flask research dashboard
named **HealthGraph Readmission Lab**.

This is a research prototype. It is not medical advice, a medical device, a
clinical decision-support system, or a clinically validated product.

---

## Table of Contents

1. [Project Identity](#project-identity)
2. [Research Problem](#research-problem)
3. [Research Objectives](#research-objectives)
4. [Principal Contributions](#principal-contributions)
5. [Dataset and Prediction Target](#dataset-and-prediction-target)
6. [End-to-End System Architecture](#end-to-end-system-architecture)
7. [Data Audit and Feature Engineering](#data-audit-and-feature-engineering)
8. [Patient-Safe Experimental Design](#patient-safe-experimental-design)
9. [Predictive Models](#predictive-models)
10. [Healthcare Graph Design](#healthcare-graph-design)
11. [Warshall Reachability Demonstration](#warshall-reachability-demonstration)
12. [GCN Robustness Simulation](#gcn-robustness-simulation)
13. [Evaluation Protocol](#evaluation-protocol)
14. [Verified Experimental Results](#verified-experimental-results)
15. [HealthGraph Readmission Lab Dashboard](#healthgraph-readmission-lab-dashboard)
16. [Repository Architecture](#repository-architecture)
17. [Installation](#installation)
18. [Running the Experiments](#running-the-experiments)
19. [Testing and Validation](#testing-and-validation)
20. [Reproducibility and Artifact Management](#reproducibility-and-artifact-management)
21. [Research Documentation](#research-documentation)
22. [Known Limitations](#known-limitations)
23. [Responsible Use](#responsible-use)
24. [Dataset Acknowledgement](#dataset-acknowledgement)
25. [GitHub Publication Guide](#github-publication-guide)
26. [License and Usage Restrictions](#license-and-usage-restrictions)
27. [Author](#author)

---

## Project Identity

### Research project title

**A Hybrid Framework of Graph Algorithms and AI for Relational Modeling of
Healthcare Data**

### Application name

**HealthGraph Readmission Lab**

### What the project does

The project models diabetes-related hospital encounters in two complementary
ways:

- as structured clinical feature vectors suitable for conventional
  machine-learning models; and
- as related entities connected through diagnosis, medication, admission, lab,
  and patient relationships.

It then investigates whether graph-aware learning provides useful predictive
information beyond tabular baselines when estimating the risk of readmission
within 30 days.

The correct interpretation is a **discharge-time risk model**. Encounter-level
treatment, utilization, admission, and discharge information may be used. The
system does not predict diabetes onset.

---

## Research Problem

The central research question is:

> Can relational information derived from shared clinical characteristics
> improve or complement conventional tabular modelling for predicting 30-day
> readmission after a diabetes-related hospital encounter?

Three related analytical challenges motivate the framework:

1. **Independent-row assumptions:** conventional models do not directly encode
   relationships among encounters.
2. **Patient leakage risk:** repeated encounters from one patient can create
   optimistic estimates if rows are split independently.
3. **Relationship uncertainty:** graph edges derived from coded healthcare data
   may be incomplete or erroneous, so robustness must be measured rather than
   assumed.

The research does not assume that model complexity determines performance.
Logistic Regression, PCA+kNN, and the GCN are evaluated on the same patient-safe
test population, and the measured metrics determine the conclusion.

---

## Research Objectives

The implemented objectives are:

1. audit and clean the source dataset using transparent, documented rules;
2. create a deterministic patient-grouped train/validation/test split;
3. build an interpretable Logistic Regression baseline;
4. evaluate an actual standardized PCA plus kNN comparison method;
5. construct a heterogeneous healthcare graph from observed dataset fields;
6. derive scalable, split-local encounter similarity graphs;
7. implement a portable two-layer GCN using pure PyTorch sparse operations;
8. demonstrate Warshall's algorithm on a small five-entity reachability matrix;
9. test GCN sensitivity to missing and noisy graph relationships;
10. evaluate all predictive methods using imbalance-aware metrics;
11. save reproducible models, predictions, curves, statistics, and
    configurations; and
12. communicate the methodology and results through a professional Flask
    dashboard.

---

## Principal Contributions

### Leakage-aware healthcare preprocessing

Identifiers remain metadata, clinical missingness is handled explicitly, rare
categories are learned from training data, and all fitting operations are
restricted to the training population.

### Patient-grouped evaluation

Every encounter belonging to one patient is assigned to exactly one split.
This grouping is shared by all three predictive models, all graph projections,
saved predictions, and dashboard case views.

### Dual graph representation

The project distinguishes between:

- an explicit heterogeneous graph for relational representation and
  exploration; and
- an encounter-level similarity projection designed for scalable GCN node
  classification.

### Portable graph neural implementation

The two-layer GCN uses PyTorch directly and does not require PyTorch Geometric.
Normalized sparse adjacency matrices make the implementation compatible with
ordinary CPU environments.

### Honest comparative evaluation

Models are ranked primarily by PR-AUC and secondarily by ROC-AUC. The framework
does not hard-code or imply that the GCN must outperform the baselines.

### Relationship robustness analysis

The trained GCN is evaluated after controlled edge removal and noise-edge
addition while clinical features and labels remain unchanged.

### Research-facing dashboard

Precomputed artifacts support interactive exploration without retraining during
ordinary page requests.

---

## Dataset and Prediction Target

### Source

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset
from the UCI Machine Learning Repository.

The source dataset contains:

- 101,766 hospital encounters;
- 71,518 unique de-identified patient identifiers in the unfiltered full file;
- demographic, admission, utilization, diagnosis, laboratory, medication, and
  readmission fields; and
- a separate mapping file for admission type, discharge disposition, and
  admission source identifiers.

### Prediction target

The original `readmitted` field is converted into a binary target:

| Original value | Binary target | Interpretation |
|---|---:|---|
| `<30` | 1 | Readmitted within 30 days |
| `>30` | 0 | Readmitted after 30 days |
| `NO` | 0 | No recorded readmission |

### Supported dataset modes

#### Reduced mode

- default execution mode;
- exactly 20,000 target-stratified source rows;
- generated with `random_state=42`;
- suitable for demonstration on an ordinary CPU; and
- recreated deterministically from the full dataset if absent.

#### Full mode

- processes the original 101,766 encounters;
- uses the same methodology and configuration structure; and
- requires substantially more time and memory, particularly for PCA, kNN
  inference, graph projection, and GCN training.

### Eligibility rule

Encounters whose mapped discharge descriptions indicate death, expiration, or
hospice care are excluded. The exclusion is based on readable mapping text,
not an unexplained list of numeric identifiers, because early readmission is not
a meaningful follow-up outcome for those encounters.

---

## End-to-End System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    UCI ENCOUNTER DATA + ID MAPPINGS                      │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA AUDIT                                                              │
│ ? → missing values · readable ID mapping · eligibility filtering        │
│ missingness report · dataset summary · target construction              │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FEATURE ENGINEERING                                                     │
│ age midpoint · ICD-9 categories · medication activity · explicit labs   │
│ train-fitted imputation · rare grouping · scaling · one-hot encoding    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ DETERMINISTIC PATIENT-GROUPED SPLIT                                     │
│                  Train 70% · Validation 15% · Test 15%                  │
│                    No patient may cross populations                      │
└───────────────┬───────────────────┴─────────────────────┬────────────────┘
                │                                         │
                ▼                                         ▼
┌──────────────────────────────┐         ┌─────────────────────────────────┐
│ TABULAR MODEL PATH           │         │ RELATIONAL MODEL PATH           │
│                              │         │                                 │
│ Logistic Regression          │         │ Heterogeneous healthcare graph  │
│                              │         │              │                  │
│ Standardization → PCA → kNN │         │ Relation-token projection       │
└───────────────┬──────────────┘         │              │                  │
                │                        │ Split-local weighted graphs     │
                │                        │              │                  │
                │                        │ Two-layer sparse PyTorch GCN    │
                │                        └──────────────┬──────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ COMMON EVALUATION                                                       │
│ PR-AUC · ROC-AUC · balanced accuracy · precision · recall · F1         │
│ confusion matrices · validation thresholds · curves · runtimes          │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│ GCN ROBUSTNESS SIMULATION    │        │ SAVED RESEARCH ARTIFACTS         │
│ edge removal + noise edges   │        │ models · metrics · predictions  │
└──────────────────────────────┘        │ graphs · histories · figures    │
                                        └────────────────┬─────────────────┘
                                                         ▼
                                        ┌──────────────────────────────────┐
                                        │ HEALTHGRAPH READMISSION LAB      │
                                        │ Flask + HTML + CSS + JavaScript │
                                        └──────────────────────────────────┘
```

### Architectural principles

- Predictive training is separate from dashboard serving.
- The Flask application reads saved artifacts and never retrains on page load.
- Paths are defined with `pathlib` for Windows portability.
- Random seeds and experiment parameters are centralized.
- Graph edges never connect different data splits.
- Browser visualization uses a manageable sample rather than the complete
  graph.
- Technical results are accompanied by plain-language limitations.

---

## Data Audit and Feature Engineering

### Identifier handling

`encounter_id` and `patient_nbr` are preserved for metadata, split assignment,
prediction joins, and dashboard lookup. They are never included as predictive
features.

### Missing values

- literal `?` values become missing values;
- numeric values receive training-set median imputation;
- general categorical missingness becomes `Unknown`;
- missing `max_glu_serum` becomes `Not Measured`;
- missing `A1Cresult` becomes `Not Measured`; and
- clinically relevant columns are retained unless an auditable reason supports
  removal.

`weight` is removed because more than 96% of eligible reduced-mode records are
missing. `payer_code` and `medical_specialty` remain available with explicit
missing categories.

### Age transformation

Ten-year age intervals are converted to ordinal numeric midpoints for modelling.
The readable age interval is retained for dashboard presentation.

Example:

```text
[60-70) → 65
[70-80) → 75
```

### ICD-9 diagnosis grouping

`diag_1`, `diag_2`, and `diag_3` are mapped into reproducible broad categories:

| ICD-9 definition | Derived category |
|---|---|
| 250.xx | Diabetes |
| 390–459 or 785 | Circulatory |
| 460–519 or 786 | Respiratory |
| 520–579 or 787 | Digestive |
| 580–629 or 788 | Genitourinary |
| 710–739 | Musculoskeletal |
| 800–999 | Injury |
| 140–239 | Neoplasm |
| V codes, E codes, remaining valid codes | Other |
| Missing or unparseable values | Unknown |

### Medication transformation

The source medication statuses are converted to compact active indicators:

```text
Up, Down, Steady → active medication (1)
No               → inactive medication (0)
```

These features support both predictive modelling and
`received_medication` graph relationships.

### Rare categories

Categorical values below 0.5% of the training population are grouped into
`Other`. The grouping threshold and retained vocabulary are learned only from
training data. Validation and test transformations cannot modify that
vocabulary.

### Serialized preprocessing

The complete Logistic Regression pipeline and PCA+kNN pipeline are serialized
with their fitted preprocessing steps. The GCN uses its own serialized
training-fitted feature transformer and relation schema.

---

## Patient-Safe Experimental Design

Repeated hospital encounters make ordinary row-level random splitting unsafe.
If the same patient appears in both training and testing, the evaluation may
benefit from patient-specific patterns that would not be available for a truly
unseen patient.

The implemented split is:

| Population | Approximate proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The grouping key is `patient_nbr`, and the random seed is 42. A patient-level
stratum records whether the patient has any positive encounter. All encounters
inherit their patient's assignment.

### Verified reduced-mode split

| Split | Encounters | Unique patients | Positive encounters | Positive prevalence |
|---|---:|---:|---:|---:|
| Training | 13,715 | 12,383 | 1,559 | 11.37% |
| Validation | 2,927 | 2,653 | 336 | 11.48% |
| Testing | 2,889 | 2,654 | 330 | 11.42% |

Verified pairwise patient overlap:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

The common assignment is saved to:

```text
data/processed/split_assignments.csv
```

---

## Predictive Models

### 1. Logistic Regression

The interpretable baseline applies training-fitted preprocessing followed by a
class-balanced Logistic Regression model.

For feature vector \(x\), the predicted probability is:

\[
P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}
\]

Implemented controls:

- class weighting addresses target imbalance;
- `C ∈ {0.1, 1.0}` is compared on validation PR-AUC;
- the verified selected value is `C=0.1`;
- the decision threshold is selected on validation data by positive-class F1;
- test metrics are reported at the selected threshold and at 0.5;
- strongest positive and negative coefficients are saved with readable names;
  and
- training and inference time are recorded.

Coefficients describe model associations, not causal clinical effects.

### 2. Standardized PCA + k-Nearest Neighbours

The second comparison method uses:

```text
training-fitted clinical preprocessing
        ↓
full dense-matrix standardization
        ↓
randomized Principal Component Analysis
        ↓
distance-weighted k-Nearest Neighbours
```

PCA is applied to a manageable dense engineered matrix:

\[
Z = XW
\]

where the columns of \(W\) are directions of decreasing variance.

Implemented controls:

- randomized PCA uses seed 42;
- a 90% explained-variance objective is evaluated within a practical
  50-component cap;
- the verified run reached the cap at 50 components and captured 57.0% of total
  standardized variance;
- the artifact explicitly records that the 90% objective was not reached;
- neighbour candidates are 7, 15, and 31;
- the verified selected value is 15;
- neighbours are weighted by distance; and
- threshold selection uses validation data only.

The variance target is a selection objective, not a guaranteed outcome when the
runtime cap is reached.

### 3. Graph Convolutional Network

The GCN treats each eligible hospital encounter as a node. Each node contains
the engineered encounter feature vector and the binary readmission label.

For a normalized adjacency matrix \(\hat{A}\), layer \(l\) computes:

\[
H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)} + b^{(l)})
\]

The adjacency is normalized as:

\[
\hat{A} = D^{-1/2}(A + I)D^{-1/2}
\]

The implemented architecture is:

```text
Input encounter features
        ↓
Graph convolution
        ↓
ReLU
        ↓
Dropout
        ↓
Graph convolution
        ↓
One binary logit per encounter
```

Training uses:

- weighted `BCEWithLogitsLoss`;
- positive-class weight derived from the training distribution;
- Adam optimization;
- weight decay;
- deterministic NumPy and PyTorch seeds;
- CPU-compatible sparse matrix multiplication;
- validation PR-AUC early stopping;
- a hidden width of 32;
- dropout of 0.35;
- learning rate of 0.01;
- maximum 100 epochs; and
- early-stopping patience of 12 epochs.

The model is saved as a state dictionary together with the architecture
configuration rather than as an unsafe opaque executable object.

---

## Healthcare Graph Design

### Heterogeneous representation

The graph uses entity types that are supported by actual dataset fields.

### Node types

- **Patient**
- **Encounter**
- **DiagnosisCategory**
- **Medication**
- **AdmissionType**
- **AdmissionSource**
- **LabResultCategory**, when a test result is available

Doctor nodes are deliberately absent because the source dataset does not contain
an individual doctor identifier.

### Edge types

| Source | Relationship | Target |
|---|---|---|
| Patient | `has_encounter` | Encounter |
| Encounter | `diagnosed_with` | DiagnosisCategory |
| Encounter | `received_medication` | Medication |
| Encounter | `admitted_as` | AdmissionType |
| Encounter | `originated_from` | AdmissionSource |
| Encounter | `has_lab_result` | LabResultCategory |

### Verified reduced-mode graph counts

| Node type | Count |
|---|---:|
| Patient | 17,690 |
| Encounter | 19,531 |
| DiagnosisCategory | 9 |
| Medication | 18 |
| AdmissionType | 8 |
| AdmissionSource | 16 |
| LabResultCategory | 6 |
| **Total** | **37,278** |

| Relationship | Edge count |
|---|---:|
| `has_encounter` | 19,531 |
| `diagnosed_with` | 44,995 |
| `received_medication` | 23,149 |
| `admitted_as` | 19,531 |
| `originated_from` | 19,531 |
| `has_lab_result` | 4,317 |
| **Total** | **131,054** |

The complete graph is represented with sparse/tabular structures to avoid
wasteful browser or NetworkX materialization. NetworkX is used for graph
statistics and a small sampled visualization.

### Encounter-level GCN projection

The GCN uses a separate encounter projection rather than directly applying
message passing across every heterogeneous entity.

Each encounter receives relation tokens for:

- broad diagnosis categories;
- active medications;
- mapped admission type; and
- mapped admission source.

A binary relation-token schema is fitted on training encounters. Within each
split, a bounded inverted index supplies candidate neighbours, and cosine
similarity is computed only for those candidates. This avoids constructing an
all-pairs \(N \times N\) similarity matrix.

Projection rules:

- request up to 8 neighbours per encounter;
- require similarity of at least 0.35 where possible;
- retain the strongest available candidate for otherwise isolated nodes;
- remove self-edges before normalization;
- symmetrize relationships;
- preserve cosine similarity as the edge weight; and
- build independent train, validation, and test graphs.

Verified test projection:

```text
Nodes:             2,889
Undirected edges: 16,524
Isolated nodes:        0
Cross-split edges:     0
```

---

## Warshall Reachability Demonstration

Warshall's algorithm is included to demonstrate indirect reachability in a
small directed graph. It is not used to create GCN predictions and is not
applied to the complete healthcare graph.

### Five entities

```text
P = Patient
D = Disease
M = Medication
L = Lab Test
C = Complication
```

### Initial adjacency matrix

| From \ To | P | D | M | L | C |
|---|---:|---:|---:|---:|---:|
| P | 0 | 1 | 0 | 1 | 0 |
| D | 0 | 0 | 1 | 0 | 1 |
| M | 0 | 0 | 0 | 0 | 0 |
| L | 0 | 1 | 0 | 0 | 0 |
| C | 0 | 0 | 0 | 0 | 0 |

### Recurrence

\[
T[i][j] = T[i][j] \lor (T[i][k] \land T[k][j])
\]

The implementation stores \(T^{(0)}\) through \(T^{(5)}\), records new pairs
at every iteration, and programmatically verifies the final closure.

Final indirect conclusions include:

- Patient reaches Disease, Medication, Lab Test, and Complication;
- Disease reaches Medication and Complication; and
- Lab Test reaches Disease, Medication, and Complication.

Complexity:

```text
Time:   O(V³)
Memory: O(V²)
```

This cost makes complete transitive closure unnecessary and infeasible for the
full healthcare graph. Detailed derivation is available in
[`docs/warshall_example.md`](docs/warshall_example.md).

---

## GCN Robustness Simulation

The relationship robustness experiment changes only the test adjacency
structure. It does not change clinical features, labels, trained model weights,
or the validation-selected threshold.

Evaluated scenarios:

1. baseline graph;
2. random removal of 5% of edges;
3. random removal of 10% of edges;
4. random removal of 20% of edges;
5. addition of 5% random noise edges; and
6. addition of 10% random noise edges.

Ten fixed trial seeds make every scenario reproducible. The baseline is retained
as a common reference, and every perturbed adjacency is renormalized before
inference. Duplicate edges and unintended self-edges are rejected.

### Verified robustness summary

| Scenario | Trials | Mean edges | Mean PR-AUC (95% t interval) | Mean ROC-AUC | Mean recall | Mean F1 | Mean probability shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 10 | 16,524 | 0.1866 [0.1866, 0.1866] | 0.6058 | 0.6394 | 0.2312 | 0.0000 |
| Remove 5% | 10 | 15,698 | 0.1878 [0.1861, 0.1894] | 0.6070 | 0.6385 | 0.2323 | 0.0066 |
| Remove 10% | 10 | 14,872 | 0.1902 [0.1884, 0.1919] | 0.6083 | 0.6364 | 0.2322 | 0.0105 |
| Remove 20% | 10 | 13,219 | 0.1901 [0.1883, 0.1919] | 0.6072 | 0.6294 | 0.2318 | 0.0166 |
| Add 5% noise | 10 | 17,350 | 0.1866 [0.1847, 0.1885] | 0.6055 | 0.6348 | 0.2320 | 0.0088 |
| Add 10% noise | 10 | 18,176 | 0.1879 [0.1857, 0.1902] | 0.6074 | 0.6367 | 0.2354 | 0.0140 |

Small non-monotonic metric differences are possible across finite random
perturbations. They are not evidence that relationship corruption improves the
model. Mean probability shift grows with perturbation magnitude, while the
ranking metrics remain within a narrow range. This experiment does not establish
robustness to clinical distribution shift.

---

## Evaluation Protocol

Every predictive model is evaluated with:

- ROC-AUC;
- average precision / PR-AUC;
- accuracy;
- balanced accuracy;
- positive-class precision;
- positive-class recall;
- positive-class F1;
- confusion matrix;
- validation-selected threshold;
- training time; and
- inference time.

ROC and precision-recall curve coordinates are saved for the test population.

Uncertainty is quantified with 1,000 paired patient-clustered bootstrap
replicates. Test patients, rather than independent encounters, are sampled with
replacement; all encounters for a sampled patient are retained, and all three
models use the identical resample in each replicate. Single-class replicates are
skipped and counted.

### Why accuracy is not the primary metric

Only about 11% of eligible reduced-mode encounters are positive. A model that
mostly predicts the negative class can appear accurate while identifying few
early readmissions. For this reason:

1. PR-AUC is the primary ranking metric;
2. ROC-AUC is the tie-breaker;
3. recall, precision, F1, and balanced accuracy are reported together; and
4. no model is described as strong solely because of raw accuracy.

### Validation and test roles

- Training data fits model parameters and preprocessing.
- Validation data selects hyperparameters, early stopping, and thresholds.
- Test data is used once for final comparison and saved case predictions.

---

## Verified Experimental Results

The verified artifact set was generated from reduced mode after eligibility
filtering:

```text
Source sample:         20,000 encounters
Eligible encounters:  19,531
Excluded encounters:     469
Eligible patients:     17,690
Positive prevalence:    11.39%
```

### Consolidated test comparison

| Rank | Model | PR-AUC | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | Threshold |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.2447 | 0.6755 | 0.7217 | 0.6265 | 0.2060 | 0.5030 | 0.2923 | 0.5339 |
| 2 | Graph Convolutional Network | 0.1866 | 0.6058 | 0.5144 | 0.5688 | 0.1411 | 0.6394 | 0.2312 | 0.4798 |
| 3 | PCA + k-Nearest Neighbours | 0.1680 | 0.5913 | 0.6033 | 0.5755 | 0.1519 | 0.5394 | 0.2370 | 0.1265 |

### Runtime summary

| Model | Training time | Inference time |
|---|---:|---:|
| Logistic Regression | 1.40 s | saved in model artifact |
| Graph Convolutional Network | 6.96 s | saved in model artifact |
| PCA + k-Nearest Neighbours | 4.27 s including sensitivity grid | saved in model artifact |

The complete forced research run took **206.44 seconds** on the verified Windows
CPU environment. Wall time varies by hardware; the split-local graph projection
was the dominant stage in this run.

### Interpretation

Logistic Regression achieved the highest PR-AUC and ROC-AUC on the patient-safe
test population. The GCN achieved the highest recall but also produced many more
false-positive predictions at its validation-selected threshold. PCA+kNN
ranked third by PR-AUC after complete dense-matrix standardization.

The experiment therefore does not support a claim that the GCN is superior to
the leading tabular baseline. The result instead shows that:

- relational modelling can be implemented and evaluated consistently;
- graph-aware learning does not automatically outperform a well-regularized
  baseline;
- limited relation diversity and engineered similarity edges may constrain GCN
  benefit;
- minority-class precision remains modest for every model; and
- the system is suitable for methodological research, not clinical use.

Complete machine-readable results are stored in:

```text
artifacts/metrics/model_comparison.json
artifacts/metrics/model_comparison.csv
```

### Patient-clustered bootstrap confidence intervals

All 1,000 requested replicates were valid; no single-class replicate was
discarded. The test set contains 2,889 encounters from 2,654 unique patients.

| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Accuracy (95% CI) | Balanced accuracy (95% CI) | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.2447 [0.2022, 0.2894] | 0.6755 [0.6409, 0.7074] | 0.7217 [0.7054, 0.7380] | 0.6265 [0.5975, 0.6552] | 0.2060 [0.1770, 0.2352] | 0.5030 [0.4479, 0.5590] | 0.2923 [0.2554, 0.3276] |
| Graph Convolutional Network | 0.1866 [0.1520, 0.2265] | 0.6058 [0.5735, 0.6392] | 0.5144 [0.4972, 0.5317] | 0.5688 [0.5423, 0.5974] | 0.1411 [0.1235, 0.1595] | 0.6394 [0.5863, 0.6918] | 0.2312 [0.2052, 0.2575] |
| PCA + kNN | 0.1680 [0.1398, 0.2055] | 0.5913 [0.5556, 0.6252] | 0.6033 [0.5859, 0.6197] | 0.5755 [0.5469, 0.6054] | 0.1519 [0.1314, 0.1728] | 0.5394 [0.4869, 0.5932] | 0.2370 [0.2077, 0.2650] |

### Paired model differences

The table reports left model minus right model. The two-sided bootstrap tail
probability is a descriptive paired-resampling measure, not proof of clinical
significance.

| Comparison | Metric | Observed difference | 95% bootstrap interval | P(difference > 0) | Two-sided tail probability |
|---|---|---:|---|---:|---:|
| Logistic Regression − GCN | PR-AUC | +0.0582 | [0.0245, 0.0926] | 1.000 | 0.002 |
| Logistic Regression − GCN | ROC-AUC | +0.0696 | [0.0335, 0.1013] | 1.000 | 0.002 |
| Logistic Regression − GCN | Recall | −0.1364 | [−0.1982, −0.0776] | 0.000 | 0.002 |
| Logistic Regression − GCN | F1 | +0.0610 | [0.0316, 0.0883] | 1.000 | 0.002 |
| Logistic Regression − PCA+kNN | PR-AUC | +0.0767 | [0.0415, 0.1074] | 1.000 | 0.002 |
| Logistic Regression − PCA+kNN | ROC-AUC | +0.0841 | [0.0465, 0.1175] | 1.000 | 0.002 |
| Logistic Regression − PCA+kNN | Recall | −0.0364 | [−0.0981, 0.0249] | 0.106 | 0.262 |
| Logistic Regression − PCA+kNN | F1 | +0.0552 | [0.0256, 0.0842] | 1.000 | 0.002 |
| GCN − PCA+kNN | PR-AUC | +0.0185 | [−0.0166, 0.0485] | 0.860 | 0.282 |
| GCN − PCA+kNN | ROC-AUC | +0.0145 | [−0.0248, 0.0557] | 0.771 | 0.460 |
| GCN − PCA+kNN | Recall | +0.1000 | [0.0344, 0.1724] | 0.999 | 0.004 |
| GCN − PCA+kNN | F1 | −0.0058 | [−0.0323, 0.0216] | 0.337 | 0.675 |

### GCN multi-seed stability

The primary comparison remains seed 42. Seeds 52, 62, 72, and 82 are a
stability analysis; no best seed is substituted into the primary comparison.

| Metric | Mean | Standard deviation | Minimum | Maximum | 95% t interval |
|---|---:|---:|---:|---:|---|
| PR-AUC | 0.1800 | 0.0107 | 0.1616 | 0.1874 | [0.1667, 0.1933] |
| ROC-AUC | 0.6019 | 0.0094 | 0.5857 | 0.6089 | [0.5902, 0.6136] |
| Balanced accuracy | 0.5664 | 0.0052 | 0.5572 | 0.5699 | [0.5599, 0.5729] |
| Precision | 0.1413 | 0.0023 | 0.1388 | 0.1451 | [0.1384, 0.1443] |
| Recall | 0.6164 | 0.0682 | 0.5242 | 0.6818 | [0.5317, 0.7011] |
| F1 | 0.2295 | 0.0037 | 0.2229 | 0.2318 | [0.2249, 0.2341] |

### PCA component sensitivity

PCA was fitted on training data only for 25, 50, 75, and 100 components. The
requested 150-component condition was invalid because the engineered training
matrix permits at most 137 components. The final choice uses validation PR-AUC,
then validation ROC-AUC, then fewer components; test results were not part of
selection.

| Components | Explained variance | Validation PR-AUC | Validation ROC-AUC | Selected k | Runtime |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.3635 | 0.1637 | 0.5980 | 31 | 1.62 s |
| **50** | **0.5700** | **0.1828** | 0.5955 | **15** | 0.61 s |
| 75 | 0.7601 | 0.1823 | 0.5862 | 31 | 0.80 s |
| 100 | 0.9284 | 0.1750 | 0.5959 | 31 | 0.86 s |

Retaining 92.84% variance with 100 components did not maximize validation
PR-AUC. The selected 50-component model retains 57.00% variance and is reported
without implying that a 90% variance threshold was required.

### Graph ablation findings

Identity-only and matched-random conditions are retrained using their own
training graphs and validation selection. Feature shuffling is an inference-only
stress test of the primary GCN. No ablation is trained or tuned on test data.

| Condition | PR-AUC | ROC-AUC | Balanced accuracy | Recall | F1 | PR-AUC difference from original |
|---|---:|---:|---:|---:|---:|---:|
| Original clinical graph | 0.1866 | 0.6058 | 0.5688 | 0.6394 | 0.2312 | 0.0000 |
| Identity-only adjacency | 0.2370 | 0.6830 | 0.6349 | 0.5879 | 0.2900 | +0.0504 |
| Matched random graph | 0.1857 | 0.6200 | 0.5801 | 0.6061 | 0.2394 | −0.0009 |
| Feature-shuffled inference | 0.1376 | 0.5319 | 0.5314 | 0.4606 | 0.2027 | −0.0490 |

The original graph does not outperform the identity-only ablation and has
nearly the same PR-AUC as the matched random graph. Consequently, this experiment
does not demonstrate a predictive benefit from the current neighbour structure.
The feature-shuffled decline shows dependence on feature alignment, but it does
not rescue the graph-construction hypothesis. The clinically constructed
similarity graph requires refinement and broader external evaluation.

---

## HealthGraph Readmission Lab Dashboard

The Flask dashboard presents saved experimental evidence. It does not retrain
models during page requests.

### Pages

| Route | Research function |
|---|---|
| `/` | Project overview, research question, workflow, and measured model summary |
| `/dataset` | Dataset modes, patient split proof, missingness, distributions, and searchable dictionary |
| `/warshall` | Synchronized \(T^{(0)}\)–\(T^{(5)}\) matrix, path calculations, and directed graph |
| `/models` | Point estimates, confidence intervals, paired differences, seed stability, PCA sensitivity, ablations, and curves |
| `/graph` | Filterable sampled heterogeneous healthcare graph |
| `/robustness` | Repeated graph perturbation means, intervals, individual trials, and probability shifts |
| `/cases` | Test encounter outcomes, probabilities, decisions, features, and neighbours |
| `/limitations` | Methodology, interpretation boundaries, and responsible-use statement |

### JSON APIs

| Endpoint | Response |
|---|---|
| `/api/health` | Application and artifact readiness |
| `/api/dataset` | Dataset audit, distributions, split summary, and dictionary |
| `/api/models` | Complete model comparison |
| `/api/statistics` | Bootstrap intervals, paired differences, and GCN seed stability |
| `/api/warshall` | Nodes, matrices, new pairs, and complexity |
| `/api/graph` | Browser-safe sampled graph |
| `/api/robustness` | Repeated perturbation summary and trial-level results |
| `/api/pca-analysis` | PCA component validation study |
| `/api/ablation` | Graph-contribution ablations |
| `/api/cases?q=<text>` | Test encounter search |
| `/api/cases/<encounter_id>` | One saved encounter case |

Reviewed CSV and JSON research tables are downloadable from
`/downloads/results/<filename>` through a strict allow-list.

### Missing-artifact behavior

If training artifacts are absent, the dashboard displays a setup screen with:

```powershell
python run_experiments.py --dataset reduced
```

No synthetic result is substituted and no model training is triggered by a web
request.

### Frontend design

The interface uses:

- semantic HTML;
- responsive CSS;
- vanilla JavaScript;
- local SVG-based charts;
- a local graph viewer;
- keyboard-friendly controls;
- loading and error states; and
- no runtime dependency on a web visualization CDN.

---

## Repository Architecture

```text
UROP/
├── app.py
│   └── Flask application factory, dashboard routes, and JSON APIs
│
├── run_experiments.py
│   └── Complete reproducible experiment orchestration
│
├── requirements.txt
├── pytest.ini
├── LICENSE
├── README.md
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   └── processed/
│       ├── diabetic_data_reduced.csv
│       └── split_assignments.csv
│
├── src/
│   └── urop_healthgraph/
│       ├── __init__.py
│       ├── config.py
│       ├── data_processing.py
│       ├── feature_engineering.py
│       ├── split_manager.py
│       ├── baseline_models.py
│       ├── graph_builder.py
│       ├── gcn_model.py
│       ├── statistical_analysis.py
│       ├── robustness.py
│       ├── graph_ablation.py
│       ├── evaluation.py
│       ├── warshall.py
│       └── artifacts.py
│
├── web/
│   ├── templates/
│   │   ├── base.html
│   │   ├── setup.html
│   │   ├── overview.html
│   │   ├── dataset.html
│   │   ├── warshall.html
│   │   ├── models.html
│   │   ├── graph.html
│   │   ├── robustness.html
│   │   ├── cases.html
│   │   └── limitations.html
│   └── static/
│       ├── css/styles.css
│       └── js/
│           ├── dashboard.js
│           ├── charts.js
│           └── graph_viewer.js
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── graphs/
│   ├── predictions/
│   └── figures/
│
├── docs/
│   ├── methodology.md
│   ├── experimental_protocol.md
│   ├── statistical_analysis.md
│   ├── robustness_protocol.md
│   ├── graph_ablation.md
│   ├── results_summary.md
│   ├── github_upload_guide.md
│   ├── reproducibility_manifest.json
│   ├── warshall_example.md
│   ├── data_dictionary.md
│   └── model_card.md
│
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_splits.py
    ├── test_warshall.py
    ├── test_graph.py
    ├── test_statistical_analysis.py
    ├── test_robustness.py
    ├── test_evaluation.py
    └── test_web.py
```

---

## Installation

### Prerequisites

- Python 3.11 or newer is recommended;
- 64-bit operating system;
- sufficient memory for the selected dataset mode; and
- CPU execution is supported.

The verified development environment used:

```text
Python 3.13.2
PyTorch 2.11.0+cpu
```

### Windows installation

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS activation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### PyTorch note

The project uses core PyTorch and does not require PyTorch Geometric. If the
generic `torch` package cannot be installed for a particular operating system or
Python version, install the appropriate CPU build using the current official
PyTorch installation selector, then install the remaining requirements.

---

## Dataset Setup

The source data is not authored by the project author and remains subject to the
UCI dataset's own terms.

Expected locations:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

Obtain both files from the official
[UCI Diabetes 130-US Hospitals dataset page](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008).

The reduced file is expected at:

```text
data/processed/diabetic_data_reduced.csv
```

If it is absent, reduced-mode execution recreates it deterministically from the
full source file.

---

## Running the Experiments

### Quick reduced experiment

```powershell
python run_experiments.py --dataset reduced --quick
```

This runs the three primary models, one GCN seed, and one robustness trial per
scenario. It omits the bootstrap and graph ablation.

### Complete research experiment

```powershell
python run_experiments.py --dataset reduced --research
```

Research mode is also the default when neither profile flag is supplied:

```powershell
python run_experiments.py --dataset reduced
```

It runs 1,000 patient-clustered bootstrap replicates, five GCN seeds, ten
robustness trials per scenario, PCA sensitivity, graph ablations, figures, and
the reproducibility manifest. Use `--force` only when every stage must be
recomputed:

```powershell
python run_experiments.py --dataset reduced --research --force
```

Bootstrap behavior may be overridden explicitly:

```powershell
python run_experiments.py --dataset reduced --bootstrap 2000
python run_experiments.py --dataset reduced --skip-bootstrap
```

### Full dataset experiment

```powershell
python run_experiments.py --dataset full --research
```

### Pipeline stages

The runner reports progress for:

1. data audit;
2. patient-grouped split creation;
3. feature engineering;
4. Warshall verification;
5. Logistic Regression training;
6. PCA component sensitivity and PCA+kNN training;
7. heterogeneous and similarity graph construction;
8. fixed GCN preprocessing;
9. primary GCN training;
10. repeated GCN seed training;
11. repeated robustness trials;
12. graph ablation;
13. patient-clustered paired bootstrap;
14. artifact and figure generation; and
15. final integrity and reproducibility checks.

Without `--force`, a complete artifact set for the requested dataset mode is
reused.

### Launch the dashboard

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Testing and Validation

Run the complete automated suite:

```powershell
python -m pytest -q
```

The tests cover:

- binary target conversion;
- deterministic reduced-dataset reproduction;
- ICD-9 category mapping;
- patient-disjoint splitting;
- training-only preprocessing behavior;
- verified Warshall closure;
- graph node and edge validity;
- absence of self-edges before normalization;
- normalized-adjacency dimensions;
- GCN forward output shape;
- evaluation metric schema;
- patient-cluster group resampling and paired-bootstrap schemas;
- safe handling of invalid single-class bootstrap samples;
- ordered confidence-interval bounds;
- repeated robustness reproducibility and output consistency;
- exact matched-random graph construction;
- six Warshall API states and newly discovered cells;
- Warshall JavaScript identifiers and step-zero initialization;
- dashboard page smoke tests; and
- missing-artifact and downloadable-result behavior.

Verified status:

```text
19 passed
```

An independent live Flask smoke test also returned HTTP 200 for all dashboard
pages, required API groups, health status, search, and a saved encounter lookup.

---

## Reproducibility and Artifact Management

### Central configuration

`src/urop_healthgraph/config.py` centralizes:

- random seed;
- path definitions;
- dataset mode;
- split ratios;
- rare-category threshold;
- graph neighbour count;
- graph similarity threshold;
- GCN hidden size;
- dropout;
- learning rate;
- weight decay;
- epoch limit;
- early-stopping patience; and
- perturbation levels.

### Saved artifacts

```text
artifacts/models/
├── logistic_regression.joblib
├── pca_knn.joblib
├── gcn_preprocessor.joblib
├── relation_schema.joblib
├── gcn_state.pt
├── gcn_seed_52.pt, gcn_seed_62.pt, gcn_seed_72.pt, gcn_seed_82.pt
├── gcn_ablation_identity.pt
└── gcn_ablation_random.pt

artifacts/metrics/
├── dataset_summary.json
├── experiment_config.json
├── model_comparison.json
├── model_comparison.csv
├── bootstrap_confidence_intervals.json/.csv
├── paired_model_differences.json/.csv
├── gcn_seed_stability.json/.csv
├── robustness_trials.json/.csv
├── robustness_summary.json/.csv
├── pca_component_analysis.json/.csv
├── graph_ablation_results.json/.csv
└── run_manifest.json

artifacts/graphs/
├── graph_statistics.json
├── sample_graph.json
└── warshall_iterations.json

artifacts/predictions/
├── test_predictions.csv
└── case_records.json

artifacts/figures/
├── model_comparison.png
└── robustness.png
```

### Test prediction schema

The common test prediction file contains:

- encounter identifier;
- patient identifier;
- actual target;
- all three model probabilities;
- all three thresholded decisions;
- split identifier;
- readable age group;
- mapped admission information;
- diagnosis categories; and
- active medications.

Case records additionally include graph neighbours and shared relation evidence.

---

## Research Documentation

Detailed supporting documents are available under `docs/`:

- [`methodology.md`](docs/methodology.md) — cleaning, feature engineering,
  modelling, graphs, and interpretation;
- [`experimental_protocol.md`](docs/experimental_protocol.md) — fixed
  parameters, ordered procedure, leakage controls, and saved evidence;
- [`statistical_analysis.md`](docs/statistical_analysis.md) — patient-clustered
  bootstrap, confidence intervals, and paired differences;
- [`robustness_protocol.md`](docs/robustness_protocol.md) — repeated graph
  perturbation design and interpretation;
- [`graph_ablation.md`](docs/graph_ablation.md) — graph-contribution study and
  measured findings;
- [`results_summary.md`](docs/results_summary.md) — final verified primary and
  secondary results;
- [`github_upload_guide.md`](docs/github_upload_guide.md) — public-release
  categories, data acquisition, and staged review;
- [`warshall_example.md`](docs/warshall_example.md) — complete worked
  transitive-closure derivation;
- [`data_dictionary.md`](docs/data_dictionary.md) — source fields and
  transformations; and
- [`model_card.md`](docs/model_card.md) — intended use, measured performance,
  limitations, and governance.

---

## Known Limitations

### Historical source period

The data represents care from 1999–2008. Clinical practice, documentation,
treatment, and hospital operations have changed since that period.

### No external validation

The models have not been evaluated on a separate hospital system, geographic
region, prospective cohort, or contemporary dataset.

### Non-causal predictions

Model coefficients, graph edges, and predicted probabilities describe
associations. They do not prove that a diagnosis, medication, or admission
characteristic causes readmission.

### Class imbalance

The positive class is uncommon. Even the best measured model has modest PR-AUC
and positive precision.

### Engineered graph relationships

Similarity edges are derived from coded relation tokens rather than direct
clinical confirmation. They may capture documentation and utilization patterns
in addition to clinical similarity.

### No hospital identifier

Patient grouping prevents patient leakage, but site-level leakage cannot be
measured because the dataset provides no hospital identifier.

### No doctor nodes

The dataset provides medical specialty but no individual clinician identifier.
Creating doctor entities would fabricate unsupported information.

### Reduced-mode trade-off

Reduced mode enables accessible CPU demonstration but does not contain every
encounter from the original dataset.

### Graph contribution

The identity-only ablation outperformed the original graph, while the matched
random graph achieved nearly identical PR-AUC. Current evidence therefore does
not establish that the engineered neighbourhood structure contributes
predictive value.

### PCA selection

The validation-selected 50-component configuration captures 57.0% variance.
Although 100 components capture 92.84%, they produce lower validation PR-AUC;
variance retention alone is not the model-selection objective.

### Clinical applicability

The framework has no prospective validation, safety analysis, regulatory
review, workflow integration study, or clinical impact assessment.

---

## Responsible Use

This project must not be used for:

- diagnosis;
- treatment recommendations;
- patient triage;
- denial or prioritization of care;
- hospital resource allocation;
- insurance or eligibility decisions;
- prediction about an identifiable person;
- deployment in a healthcare environment; or
- any clinical or operational decision.

The dashboard contains de-identified dataset references and does not fabricate
patient names. Nevertheless, healthcare data and predictive outputs must be
handled carefully and according to the source dataset's terms and applicable
institutional requirements.

---

## Dataset Acknowledgement

The project uses:

> Clore, J., Cios, K., DeShazo, J., and Strack, B.  
> *Diabetes 130-US Hospitals for Years 1999–2008*.  
> UCI Machine Learning Repository.  
> DOI: `10.24432/C5230J`

Official dataset page:

<https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008>

The dataset and mapping files are third-party materials. Their inclusion or
local use does not transfer ownership to the project author. Verify the current
UCI terms before downloading, using, or redistributing those files.

---

## GitHub Publication Guide

A public research repository should contain the implementation and authored
technical documentation while excluding local environments, row-level data,
large generated binaries, private drafts, and downloaded third-party papers.

### Commit these project files

```text
LICENSE
README.md
.gitignore
requirements.txt
pytest.ini
app.py
run_experiments.py
src/urop_healthgraph/
web/templates/
web/static/
tests/
data/README.md
docs/methodology.md
docs/experimental_protocol.md
docs/warshall_example.md
docs/data_dictionary.md
docs/model_card.md
docs/statistical_analysis.md
docs/robustness_protocol.md
docs/graph_ablation.md
docs/results_summary.md
docs/github_upload_guide.md
docs/reproducibility_manifest.json
artifacts/metrics/
artifacts/graphs/graph_statistics.json
artifacts/graphs/warshall_iterations.json
artifacts/figures/
artifacts/*/.gitkeep
```

These files contain the reproducible implementation, dependency declaration,
tests, dashboard, authored research documentation, and empty artifact-directory
markers.

### Do not commit these files to a public repository

```text
.venv/
venv/
__pycache__/
.pytest_cache/
.test-tmp/
.vscode/

data/raw/*.csv
data/processed/*.csv

artifacts/models/*
artifacts/predictions/*
artifacts/graphs/sample_graph.json

docs/*.docx
docs/*.pptx
docs/References/*.pdf
```

Reasons:

- raw and processed healthcare CSVs are third-party or row-level data and should
  be obtained directly from the official source;
- split assignments, case records, and predictions expose row-level
  de-identified references and are unnecessary in a public source release;
- serialized model files are generated binaries and may be platform-dependent;
- sampled graph JSON may retain encounter references;
- virtual environments and caches are machine-specific;
- `.vscode` contains local editor configuration;
- Office files may contain drafts, revision metadata, personal contact details,
  or material not intended for public release; and
- downloaded papers remain third-party copyrighted publications and should be
  linked through their publishers rather than redistributed.

### Optional aggregate research evidence

The following generated files contain aggregate results and may be published if
they have been reviewed manually:

```text
artifacts/metrics/dataset_summary.json
artifacts/metrics/experiment_config.json
artifacts/metrics/model_comparison.json
artifacts/metrics/model_comparison.csv
artifacts/metrics/robustness_results.json
artifacts/metrics/robustness_results.csv
artifacts/metrics/bootstrap_confidence_intervals.json
artifacts/metrics/bootstrap_confidence_intervals.csv
artifacts/metrics/paired_model_differences.json
artifacts/metrics/paired_model_differences.csv
artifacts/metrics/gcn_seed_stability.json
artifacts/metrics/gcn_seed_stability.csv
artifacts/metrics/robustness_trials.json
artifacts/metrics/robustness_trials.csv
artifacts/metrics/robustness_summary.json
artifacts/metrics/robustness_summary.csv
artifacts/metrics/pca_component_analysis.json
artifacts/metrics/pca_component_analysis.csv
artifacts/metrics/graph_ablation_results.json
artifacts/metrics/graph_ablation_results.csv
artifacts/metrics/run_manifest.json
artifacts/graphs/graph_statistics.json
artifacts/graphs/warshall_iterations.json
artifacts/figures/model_comparison.png
artifacts/figures/robustness.png
```

The `.gitignore` allows these aggregate results while excluding model binaries,
row-level predictions, the sampled encounter graph, and datasets. Review every
aggregate file before staging it.

### Safe initial staging example

Do not begin with `git add .` while local data, Office documents, and reference
PDFs are present. Stage the intended public files explicitly:

```powershell
git add LICENSE README.md .gitignore requirements.txt pytest.ini
git add app.py run_experiments.py
git add src web tests
git add data/README.md
git add docs/*.md docs/reproducibility_manifest.json
git add artifacts/metrics artifacts/figures
git add artifacts/graphs/graph_statistics.json
git add artifacts/graphs/warshall_iterations.json
git add artifacts/models/.gitkeep artifacts/predictions/.gitkeep
```

Before committing, inspect the exact staged set:

```powershell
git status --short
git diff --cached --stat
git diff --cached
```

If the repository must be publicly visible for judging, publish only the
reviewed source set above. If the intention is genuinely view-only with stronger
control over copying and forking, use a private repository and grant temporary
read access to evaluators.

---

## License and Usage Restrictions

**Copyright © 2026 Yatharth Garg. All rights reserved.**

This repository is **not open-source software**. Its source is visible solely
for private educational inspection and authorized academic or competition
evaluation.

Unless Yatharth Garg provides prior explicit written permission, you may not:

- copy or reproduce the implementation;
- use or execute it outside the narrow authorized-evaluation exception;
- modify or create derivative work from it;
- redistribute, mirror, publish, or sublicense it;
- submit it, in whole or in part, as another academic or competition project;
- sell, monetize, host, commercialize, or provide it as a service;
- incorporate its code, documentation, diagrams, interface, models, artifacts,
  or figures into another work; or
- deploy or apply it in healthcare, research operations, education delivery,
  commercial activity, or any real-world decision-making environment.

General research ideas and principles may be studied for educational purposes.
Any independent work inspired by those abstract ideas must be written and
implemented independently and must not copy the protected expression of this
project. Academic discussion should clearly attribute the project and author.

Academic supervisors, examiners, and competition or hackathon judges may run one
temporary unmodified copy solely to evaluate the author's submission. That
exception does not permit reuse, redistribution, publication, modification, or
deployment.

See the complete [`LICENSE`](LICENSE) file for the controlling terms.

If strict prevention of copying or forking is required, keep the repository
private and grant read access only to authorized evaluators. Public GitHub
repositories remain subject to GitHub's platform terms.

---

## Author

**Yatharth Garg**

Research project: **A Hybrid Framework of Graph Algorithms and AI for Relational
Modeling of Healthcare Data**
