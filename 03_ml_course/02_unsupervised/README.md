# Unsupervised learning in C++17

## What this track teaches

Unsupervised learning finds structure in rows that carry no target column. This track covers three
families: **clustering** (k-means, agglomerative/hierarchical, DBSCAN, Gaussian mixtures), **cluster
evaluation** (silhouette, inertia, BIC, subsample stability) and **dimensionality reduction** (PCA,
t-SNE). Every algorithm is written from scratch in `helper/`-backed C++17 with `fit`, `save` and
`load`, plus `predict`/`transform` where the method mathematically supports new rows.

Recommended order: k-means, hierarchical, cluster evaluation, then DBSCAN, Gaussian mixtures, PCA and
t-SNE. Isolation Forest, LOF and One-class SVM continue in `../03_anomaly_detection/`, which reuses
this track's `workflow.hpp`, `validation.hpp`, `predict_input.hpp` and test binaries.

## Dataset and why

The end-to-end projects use **Palmer Penguins** (`helper/data/penguins.csv`, 344 rows, four numeric
measurements: bill length, bill depth, flipper length, body mass). It is small enough to inspect by
hand, has three real groups (species) that are *not* given to the algorithm, contains missing
values (so imputation must be learned) and mixes units (millimetres versus grams) so standardization
visibly matters. Species labels are used only for EDA colouring and for a descriptive external
comparison *after* model selection. The k-means, hierarchical, PCA and t-SNE implementation lessons
also use Iris (`helper/data/iris.csv`) for a smaller worked example.

## Prerequisites

C++17 (vectors, lambdas, `std::function`), Euclidean distance, means/variances, and the
`helper/` toolbox (`Mat`/`Vec`, `Preprocessor`, `Artifacts`, `archive`, `Plot`); see
`../helper/README.md`. Read `01_supervised/` first if you have not met train/test splits.

## Modules and shared files

| Path | Target prefix | Method | Predict tool |
|---|---|---|---|
| `01_kmeans/` | `ukm_` | Lloyd's k-means with k-means++ seeding, k chosen by silhouette | `ukm_predict` (cluster id) |
| `02_agglomerative/` | `uag_` | Bottom-up hierarchical clustering, single/complete/average/Ward linkage, dendrogram | none (see module README) |
| `03_cluster_evaluation/` | `uce_` | Silhouette, inertia and stability used to compare k-means with Ward on identical rows | none |
| `04_dbscan/` | `udb_` | Density-based clustering with noise label -1, eps/min_samples sweep | none |
| `05_gaussian_mixture/` | `ugm_` | Diagonal Gaussian mixture fitted by EM, components chosen by BIC | `ugm_predict` (responsibilities) |
| `06_pca/` | `upca_` | Principal component analysis with a 95% explained-variance rule and hold-out reconstruction error | `upca_predict` (PC scores) |
| `07_tsne/` | `utsne_` | t-SNE embedding, perplexity chosen by trustworthiness | none |
| `workflow.hpp` | header | Shared end-to-end stages: `load_penguins`, `manifest`, `eda`, `report`, `stability`, `snapshot`, `verify` | - |
| `validation.hpp` | header | `validate_dense` (rejects empty/ragged/non-finite matrices) and `distance2` | - |
| `predict_input.hpp` | header | `read_numeric_rows` (header-less CSV, blanks/NA -> NaN) and `write_predictions` used by every `*_predict` | - |
| `tests/test_unsupervised.cpp` | `unsupervised_tests` | Numerical unit checks for every model in this track and the anomaly track (`ctest -R unsupervised_numerical`) | - |
| `tests/verify_reload.cpp` | `unsupervised_reload` | Fresh-process reload of a saved run, compared with the run's own outputs (`*_fresh_reload` tests) | - |
| `tests/penguin_inference.csv` | data | Header-less rows fed to the `*_new_rows` tests | - |

## Build and run

From the repository root, with `D:\msys64\ucrt64\bin` on `PATH`:

```powershell
cmake --preset course
cmake --build --preset course --target ukm_end_to_end ukm_predict unsupervised_tests unsupervised_reload
build\03_ml_course\02_unsupervised\01_kmeans\ukm_end_to_end.exe
build\03_ml_course\02_unsupervised\01_kmeans\ukm_predict.exe 03_ml_course\02_unsupervised\tests\penguin_inference.csv
```

Replace `ukm` by the prefix of any module. The unsupervised end-to-end programs take no flags (there
is no `--quick`: a full run takes seconds). `CMake` compiles the absolute data and output paths into
each executable, so the working directory does not matter.

## How the end-to-end projects are organised

Clustering projects (`ukm`, `uag`, `uce`, `udb`, `ugm`) are exploratory full-data analyses: they
impute and standardize all rows, select hyper-parameters with an *internal* metric (silhouette or
BIC), evaluate the chosen fit, then refit on five seeded 80% subsamples and report the adjusted Rand
index against the full-data labels as a stability check. They are not hold-out predictive estimates.
PCA uses a proper 80/20 split with preprocessing fitted on the training rows only, and t-SNE selects
its perplexity by trustworthiness. Every project ends by saving its state under `model/` and
reloading it in the same process to prove the saved file reproduces the results.

## Results layout

Every lesson writes only inside its own module: `results/<cpp-stem>_results/`. The end-to-end run
lives in `results/04_end_to_end_results/` with

- `run_manifest.json`, `report.md`, `execution.log` - configuration, interpretation and the log;
- `data/` - row bookkeeping (partition or assignment tables);
- `eda/` - descriptive statistics, histograms and correlations;
- `validation/` - candidate scores, the selected parameters and `stability.csv`;
- `evaluation/` - assignments/projections/embedding, `metrics.json`, `cluster_profiles.csv` and `figures/*.svg`;
- `model/` - `model_state.txt`, `preprocessing_state.txt`, `feature_schema.csv`;
- `inference/reload_verification.json` - the in-process reload check.

`*_predict` programs write `results/predict_results/new_predictions.csv`. The `*_fresh_reload`
tests write `results/verify_reload_results/fresh_process_verification.json`. Theory and
math-intuition lessons only print.

## Tests

```powershell
ctest --preset course -R unsupervised_numerical      # unit fixtures for every model
ctest --preset course -R "ukm|uag|udb|ugm|upca|utsne" # workflow, fresh reload and new-row tests
```

`<prefix>_workflow` runs the end-to-end program, `<prefix>_fresh_reload` reloads its saved model in a
separate process and compares outputs, and `<prefix>_new_rows` (where a predict tool exists) scores
`tests/penguin_inference.csv`. No Python runtime or reference library is involved.

## Key takeaways

- Standardize before any distance-based method; otherwise body mass in grams dominates.
- Choose k or eps with internal metrics and confirm with subsample stability, never with the labels.
- Only some methods define an out-of-sample operation: centroid assignment, mixture
  responsibilities and PCA projection do; hierarchical trees, DBSCAN and t-SNE do not.
- Saving the preprocessor with the model is what makes inference on new rows reproducible.

## Next

`../03_anomaly_detection/` applies the same workflow to unsupervised outlier scoring with
validation-calibrated thresholds.
