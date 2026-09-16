# Unsupervised learning in C++17

## What this track teaches

Unsupervised learning finds structure in rows that carry no target column. This track covers three families: clustering (k-means, agglomerative/hierarchical, DBSCAN, Gaussian mixtures), cluster evaluation (silhouette, adjusted Rand index, subsample stability) and dimensionality reduction (PCA, t-SNE). Every algorithm is written from scratch in `helper/`-backed C++17 with `fit`, `save` and `load`, plus `predict`/`transform` where the method mathematically supports new rows.

Recommended order: k-means, hierarchical, cluster evaluation, then DBSCAN, Gaussian mixtures, PCA and t-SNE. Isolation Forest, LOF and One-class SVM continue in `../03_anomaly_detection/`, which reuses this track's `workflow.hpp`, `validation.hpp`, `predict_input.hpp` and test binaries.

## Dataset and why

The end-to-end projects use Palmer Penguins (`helper/data/penguins.csv`, 344 rows, four numeric measurements: bill length, bill depth, flipper length, body mass). It is small enough to inspect by hand, has three real groups (species) that are not given to the algorithm, contains missing values (so imputation must be learned) and mixes units (millimetres versus grams) so standardization visibly matters. Species labels are used only for EDA colouring and for a descriptive external comparison after model selection. The k-means, hierarchical, PCA and t-SNE implementation lessons also use Iris (`helper/data/iris.csv`) for a smaller worked example.

## Prerequisites

C++17 (vectors, lambdas, `std::function`), Euclidean distance, means/variances, and the `helper/` toolbox (`Mat`/`Vec`, `Preprocessor`, `Artifacts`, `archive`, `Plot`). See `../helper/README.md`. Read `01_supervised/` first if you have not met training, prediction and evaluation.

## Modules

| Module | Target prefix | What it teaches | `predict` |
|---|---|---|---|
| [`01_kmeans`](01_kmeans/README.md) | `ukm_*` | k-means++ seeding, Lloyd iterations, choosing k | yes |
| [`02_agglomerative`](02_agglomerative/README.md) | `uag_*` | Linkage variants and the dendrogram | no |
| [`03_cluster_evaluation`](03_cluster_evaluation/README.md) | `uce_*` | Silhouette, adjusted Rand index and subsample stability | no |
| [`04_dbscan`](04_dbscan/README.md) | `udb_*` | Density-based clusters and noise | no |
| [`05_gaussian_mixture`](05_gaussian_mixture/README.md) | `ugm_*` | EM with soft responsibilities, choosing k by BIC | yes |
| [`06_pca`](06_pca/README.md) | `upca_*` | Eigen-decomposition, projection and reconstruction | yes |
| [`07_tsne`](07_tsne/README.md) | `utsne_*` | Perplexity-based 2-D embeddings for visualisation | no |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target ukm_end_to_end ukm_predict
build\03_ml_course\02_unsupervised\01_kmeans\ukm_end_to_end            # full run, no --quick flag
build\03_ml_course\02_unsupervised\01_kmeans\ukm_predict <numeric_csv>  # uses results\04_end_to_end_results\model
```

Unsupervised end-to-end programs take no `--quick` flag, because a full run on 344 rows finishes in seconds. Replace `ukm` with the prefix of any module. The `<prefix>_predict` programs read a header-less numeric CSV, restore the saved preprocessing and model, and write predictions to `results/predict_results/`. `03_ml_course/02_unsupervised/tests/penguin_inference.csv` is a ready-made example. Each module README documents its own input format.

## How the end-to-end projects are organised

Every end-to-end program uses the shared `workflow.hpp` and `validation.hpp`: it loads the raw CSV, imputes missing values with training medians, standardises with the `Preprocessor`, evaluates candidates on an internal metric (silhouette, BIC, explained variance or trustworthiness), fits the chosen model, saves the model, preprocessing and `feature_schema.csv`, then verifies a reload round trip. There is no target, so the fitting rows are validated internally rather than on a holdout. PCA is the one exception, with a real 80/20 train/test split and a hold-out reconstruction error. Qualities live in `results/<source stem>_results/` and diagnostics in `report.md`.

## Results layout

Each executable owns `results/<source stem>_results/` inside its own module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/04_end_to_end_results/
  data/          preprocessing notes and row sources
  eda/           per-feature summaries and figures on the full data
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selected_parameters.json
                 selection_rule.md (PCA, t-SNE), stability.csv (clustering)
  evaluation/    metrics.json, per-method tables, figures/
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
results/predict_results/new_predictions.csv  (by <prefix>_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R ukm
```

Each module registers its unit checks in `tests/test_unsupervised.cpp` (CTest entry `unsupervised_numerical`). Modules with an end-to-end stage also register `<prefix>_workflow` and `<prefix>_fresh_reload`. Modules with a `predict` target add `<prefix>_new_rows`. `03_cluster_evaluation` registers none of the integration entries, so run its executable directly. Any failed check throws, so the exit code is non-zero.

## Key takeaways

- Without a target there is no holdout. Validate internally (silhouette, inertia, BIC) and by stability.
- Standardization decides which columns dominate distance. It is not optional here.
- Prefer methods with `save`/`load` and a `predict` path when new rows will arrive.

## Next module

`../03_anomaly_detection/` for Isolation Forest, LOF and One-class SVM.