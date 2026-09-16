# Gaussian Mixture Model

## Problem and core idea

A Gaussian mixture models the data as a weighted sum of `k` Gaussians and assigns each row a soft responsibility, the probability that it came from each component. That is richer than a partition: one row contributes to every cluster with a weight. The parameters (means, covariances, mixing weights) are fit by expectation-maximization. A small variance floor on the covariance diagonals keeps the fit away from singular matrices. The number of components `k` is chosen with BIC, which trades a better fit against the number of free parameters.

## Dataset and why

Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns, 3 species) in the end-to-end project. The three species have genuinely different spreads, so soft responsibilities are informative, and BIC lands on or near `k = 3` when standardization and imputation are done right. The implementation lesson uses a 2-D fixture you inspect by eye. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_kmeans` (the comparison model), multivariate Gaussians from `01_supervised/02_classification/11_lda`, and likelihood reasoning. Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | One EM step and the responsibilities, by hand | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `GaussianMixture.hpp` | interface library `ml_gmm` | Header-only `ml::GaussianMixture`: diagonal-covariance EM with a variance floor, `fit`, `predict_proba`, `predict`, `score`, `bic`, `aic`, save/load | - |
| `01_theory.cpp` | `ugm_theory` | The mixture density, responsibilities, EM | prints only |
| `02_math_intuition.cpp` | `ugm_math_intuition` | The update equations and BIC/AIC parameter counts, by hand | prints only |
| `03_implementation.cpp` | `ugm_implementation` | Four points in two components, then three 2-D blobs of 60 points, k = 3: the EM log-likelihood trajectory over 40 refits, then responsibilities per row | `results/03_implementation_results/`: `log_likelihood.csv`, `responsibilities.csv`, `figures/log_likelihood.svg`, `figures/components_scatter.svg` |
| `04_end_to_end.cpp` | `ugm_end_to_end` | Full project on penguins: impute, standardise, fit 15 EM runs (k in {2, 3, 4, 5, 6} x seeds 42, 43, 44, up to 200 iterations), pick `k` and seed by BIC, evaluate, stability and reload-check | `results/04_end_to_end_results/` (see below) |
| `predict.cpp` | `ugm_predict` | Read a header-less numeric CSV, restore preprocessing and mixture, emit `output_0`..`output_{k-1}` as the responsibilities | `results/predict_results/new_predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Interface library `ml_gmm` headers only | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target ugm_end_to_end ugm_predict
build\03_ml_course\02_unsupervised\05_gaussian_mixture\ugm_end_to_end            # full run, no --quick flag
build\03_ml_course\02_unsupervised\05_gaussian_mixture\ugm_predict <numeric_csv> # reloads results\04_end_to_end_results\model
```

The end-to-end program has no `--quick` flag, because a full run on 344 rows takes seconds. `ugm_predict <numeric_csv>` expects a header-less CSV with the four features in the order of `results/04_end_to_end_results/model/feature_schema.csv`, in raw units. Blank, `NA` or `nan` fields are imputed with the saved training medians. The output columns `output_0`..`output_{k-1}` are the responsibilities, one per component. They sum to one per row, and the largest names the assigned cluster. `03_ml_course/02_unsupervised/tests/penguin_inference.csv` is a ready-made example. Every lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row sources and preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selected_parameters.json, stability.csv
  evaluation/    metrics.json, responsibilities.csv, cluster_profiles.csv
                 figures/ (silhouette, silhouette_distribution, clusters)
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
results/predict_results/new_predictions.csv       (written by ugm_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R ugm
```

`ugm_workflow` runs `ugm_end_to_end` and fails if any stage or the reload verification fails. `ugm_fresh_reload` starts a new process, loads the saved mixture and preprocessor, recomputes the arg-max labels and matches them against `evaluation/assignments.csv`. `ugm_new_rows` runs `ugm_predict` on `tests/penguin_inference.csv` and must exit cleanly. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`.

## Key takeaways

- Responsibilities are the honest output. The argmax cluster is a lossy projection of them.
- BIC chooses `k` without a target. Read `validation/candidate_scores.csv` for the components/seeds sweep.
- The variance floor is what keeps EM from collapsing onto a single point.

## Next module

`06_pca`: compressing the features themselves.