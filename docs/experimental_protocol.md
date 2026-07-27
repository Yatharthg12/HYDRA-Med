# Experimental Protocol

## Reproduction command

From the repository root:

```powershell
python run_experiments.py --dataset reduced --research --force
```

Use `--dataset full` for all source encounters. Full mode is implemented but was
not executed for the committed verification because its PCA, kNN inference, and
graph workload is materially larger.

## Fixed configuration

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Reduced rows | 20,000 |
| Split ratios | 70% / 15% / 15% |
| Rare-category threshold | 0.5% of training rows |
| Graph neighbours | 8 |
| Similarity threshold | 0.35 |
| GCN hidden width | 32 |
| GCN dropout | 0.35 |
| GCN learning rate | 0.01 |
| Weight decay | 0.0005 |
| Maximum epochs | 100 |
| Early-stopping patience | 12 |
| Bootstrap master seed | 42042 |
| Bootstrap replicates | 1,000 |
| GCN stability seeds | 42, 52, 62, 72, 82 |
| Robustness seeds | 42 through 51 |
| PCA component grid | 25, 50, 75, 100, 150 if valid |
| Graph-ablation seed | 4242 |

The full serialized configuration is
`artifacts/metrics/experiment_config.json`.

## Ordered pipeline

1. Load and audit the selected dataset.
2. Parse mappings and exclude mapped end-of-life/hospice encounters.
3. Create the deterministic patient-grouped split and assert disjointness.
4. Derive clinical feature frames.
5. Verify the five-node Warshall closure.
6. Tune/train Logistic Regression on train/validation.
7. Evaluate the valid PCA component grid and tune/train distance-weighted kNN
   using validation performance.
8. Count the heterogeneous graph and save its sampled view.
9. Fit the training relation schema and construct three separate similarity
   graphs.
10. Train the two-layer GCN with validation early stopping.
11. Repeat GCN training across five prespecified seeds without best-seed
    substitution.
12. Evaluate six graph-robustness scenarios across ten trial seeds.
13. Run original, identity-only, matched-random, and feature-shuffled graph
    ablations.
14. Run the paired patient-clustered bootstrap.
15. Save predictions, metrics, curves, figures, configurations, hashes, and
    model objects; then run final integrity checks.

## Leakage controls

- Predictive IDs are excluded.
- All encounters belonging to one patient share one split.
- Imputation, scaling, rare grouping, one-hot vocabularies, PCA, and relation
  vocabulary are training-fitted.
- Hyperparameters and thresholds use validation results only.
- Graph edges are constructed separately inside train, validation, and test.
- Test labels never select hyperparameters, epochs, or thresholds.
- Robustness changes only test adjacency relationships after training.
- Bootstrap draws complete patient clusters and uses one paired draw across
  every model.
- PCA count and kNN settings are selected without test results.
- Identity and random ablation models select checkpoints and thresholds on
  matching validation graphs.
- Feature-shuffled ablation is inference-only and never tunes on test labels.

## Selection rules

Logistic `C` and kNN neighbour count maximize validation PR-AUC. The decision
threshold for each model maximizes positive-class validation F1. The GCN
checkpoint maximizes validation PR-AUC with patience-based early stopping.

Final model ranking uses test PR-AUC, with test ROC-AUC as a tie-breaker. The
rule is specified before inspecting which model wins.

## Saved evidence

- Split: `data/processed/split_assignments.csv`
- Metrics: `artifacts/metrics/model_comparison.{json,csv}`
- Robustness: `artifacts/metrics/robustness_results.{json,csv}`
- Bootstrap: `artifacts/metrics/bootstrap_confidence_intervals.{json,csv}`
- Paired differences: `artifacts/metrics/paired_model_differences.{json,csv}`
- GCN stability: `artifacts/metrics/gcn_seed_stability.{json,csv}`
- Robustness trials/summary: `artifacts/metrics/robustness_{trials,summary}.{json,csv}`
- PCA analysis: `artifacts/metrics/pca_component_analysis.{json,csv}`
- Graph ablation: `artifacts/metrics/graph_ablation_results.{json,csv}`
- Curves and training histories: embedded in model comparison JSON
- Predictions: `artifacts/predictions/test_predictions.csv`
- Models: `artifacts/models/`
- Graph configuration/statistics: `artifacts/graphs/graph_statistics.json`
- Run completion proof: `artifacts/metrics/run_manifest.json`

## Validation checklist

```powershell
python -m pytest -q
python app.py
```

The 19 tests cover target conversion, deterministic sampling, ICD mapping,
training-only preprocessing, patient disjointness, grouped bootstrap behavior,
paired schemas, invalid resamples, interval ordering, repeated robustness,
matched-random graphs, all six Warshall states, dashboard initialization,
routes, APIs, downloads, and missing-artifact behavior.
