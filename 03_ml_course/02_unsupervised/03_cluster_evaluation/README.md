# Cluster Evaluation

## Problem and core idea

Clustering has no target, so there is no holdout to score against. You judge a partition by internal criteria: inertia (total within-cluster distance), silhouette (how separated clusters are), BIC (for a Gaussian mixture) and subsample stability (does the partition survive refitting on half the data?). The external criterion, the adjusted Rand index against species labels, is used only for a descriptive comparison after model selection, never to pick `k`. The metric helpers are model-agnostic: they take a label vector and produce a number.

## Dataset and why

Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns, 3 real species). The species are hidden during selection, so the module demonstrates the honest workflow: choose `k` and the algorithm by internal metrics, then compare with the external labels last. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_kmeans` (the candidate partitions) and `helper/math/metrics.hpp` for the silhouette. Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `helper/math/metrics.hpp` | - | `ml::silhouette` and `ml::silhouette_samples`, shared across the track | - |
| `workflow.hpp` | - | `ml::unsup::valid_silhouette`, `ml::unsup::adjusted_rand` (ARI) plus the manifest/report/stability/snapshot/verify helpers | - |
| `01_theory.cpp` | `uce_theory` | Internal versus external criteria, why there is no test set | prints only |
| `02_math_intuition.cpp` | `uce_math_intuition` | Silhouette on the four-point fixture, by hand (mean 47/63) | prints only |
| `03_implementation.cpp` | `uce_implementation` | Three synthetic blobs clustered by k-means for k = 2..6: silhouette curve, per-row silhouette samples, best k | `results/03_implementation_results/`: `silhouette_per_k.csv`, `silhouette_samples.csv`, `figures/silhouette_per_k.svg`, `figures/clusters_best_k.svg` |
| `04_end_to_end.cpp` | `uce_end_to_end` | Full project on penguins: impute, standardise, compare k-means and Ward over `k` in {2, 3, 4, 5, 6} by silhouette, pick `k` by the best k-means silhouette, then report the adjusted Rand index against species inside `metrics.json` as a descriptive check | `results/04_end_to_end_results/` (see below) |
| `CMakeLists.txt` | - | Registers the targets above. The module links the existing `ml_kmeans` and `ml_aggl` libraries and the shared helpers | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target uce_end_to_end
build\03_ml_course\02_unsupervised\03_cluster_evaluation\uce_end_to_end            # full run, no --quick flag
```

The end-to-end program has no `--quick` flag, because a full run on 344 rows takes seconds. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row sources and preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, stability.csv
  evaluation/    metrics.json, assignments.csv, cluster_profiles.csv, figures/
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
```

`evaluation/figures/` holds `comparison.svg` (silhouette versus k for both algorithms) plus the shared `clusters.svg`, `silhouette.svg` and `silhouette_distribution.svg`. The external ARI lives in `evaluation/metrics.json`.

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R uce
```

The module registers no dedicated CTest entry. The unit checks in `../tests/test_unsupervised.cpp` (silhouette values, ARI properties) run under `unsupervised_numerical`. `uce_workflow` is not registered, so run `uce_end_to_end` directly. There is no `predict` target and no reload CTest because the winning k-means fit is recreated from the raw CSV.

## Key takeaways

- Internal metrics pick the model. External labels only describe it afterwards.
- Silhouette and stability disagree in interesting ways. Read both.
- Every metric lives over a label vector, so they work for any clustering algorithm.

## Next module

`04_dbscan`: clusters with no `k` and no fixed shape.