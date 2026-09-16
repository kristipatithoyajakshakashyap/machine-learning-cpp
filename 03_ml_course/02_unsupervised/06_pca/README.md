# Principal Component Analysis

## Problem and core idea

PCA re-expresses the data in the directions of largest variance. It builds the covariance matrix of the standardized features, computes its eigen-decomposition, and keeps the top `k` eigenvectors as components. `transform` projects rows onto those components, `inverse_transform` reconstructs an approximation, and the explained-variance ratio tells you how much of the spread `k` components capture. PCA has a true out-of-sample `transform`, so a fitted model applies to new rows cheaply. The implementation is header-only and diagonalises the covariance matrix with cyclic Jacobi rotations, which is exact for the tiny `p` used here and deterministic, so a fixed seed is never needed.

## Dataset and why

Iris (`helper/data/iris.csv`, 150 rows, 4 columns) and Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns) in the end-to-end project. Both have four features whose pairwise correlations are strong enough that two components recover most of the variance. Species labels are hidden during the fit and used only to colour the 2-D scatter afterwards, so you see whether the projection preserves the natural groups. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

Eigen-values and eigen-vectors, covariance matrices, and `01_kmeans` (the projection is fed to it in the end-to-end project). Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | A 2-D rotation and projection, by hand | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `PCA.hpp` | interface library `ml_pca` | Header-only `ml::PCA`: covariance eigen-decomposition by cyclic Jacobi rotations, `fit`, `transform`, `inverse_transform`, `explained_variance_ratio`, `components`, save/load | - |
| `01_theory.cpp` | `upca_theory` | Variance, covariance eigen-values, projections | prints only |
| `02_math_intuition.cpp` | `upca_math_intuition` | A 2-D eigen-decomposition, projection and reconstruction, by hand | prints only |
| `03_implementation.cpp` | `upca_implementation` | Four collinear points collapse to one component, then iris in raw units projected to two: loadings, ratios, score scatter | `results/03_implementation_results/`: `explained_variance.csv`, `scores.csv`, `components.csv`, `figures/scree.svg`, `figures/scores_scatter.svg` |
| `04_end_to_end.cpp` | `upca_end_to_end` | Full project on penguins with a train/test split: impute and standardise on 80% of the rows, choose the dimension by the fixed 95% training-variance rule, report hold-out reconstruction error, snapshot and reload-check (there is no `--quick` flag) | `results/04_end_to_end_results/` (see below) |
| `predict.cpp` | `upca_predict` | Read a header-less numeric CSV, restore preprocessing and components, emit `output_0`..`output_{k-1}` as the component scores | `results/predict_results/new_predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Interface library `ml_pca` headers only | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target upca_end_to_end upca_predict
build\03_ml_course\02_unsupervised\06_pca\upca_end_to_end            # full run, no --quick flag
build\03_ml_course\02_unsupervised\06_pca\upca_predict <numeric_csv>  # reloads results\04_end_to_end_results\model
```

The end-to-end program has no `--quick` flag, because a full run on 344 rows takes seconds. `upca_predict <numeric_csv>` expects a header-less CSV with the four features in the order of `results/04_end_to_end_results/model/feature_schema.csv`, in raw units. Blank, `NA` or `nan` fields are imputed with the saved training medians. The output columns `output_0`..`output_{k-1}` are the component scores. `03_ml_course/02_unsupervised/tests/penguin_inference.csv` is a ready-made example. Every lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row_assignments.csv, preprocessing notes
  eda/           per-feature summaries and figures (training rows only)
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selection_rule.md
  evaluation/    projections.csv, metrics.json, figures/explained_variance.svg
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
results/predict_results/new_predictions.csv    (written by upca_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R upca
```

`upca_workflow` runs `upca_end_to_end` and fails if any stage or the reload verification fails. `upca_fresh_reload` starts a new process, loads the saved preprocessor and components, recomputes the test-row projections and matches them against `evaluation/projections.csv`. `upca_new_rows` runs `upca_predict` on `tests/penguin_inference.csv` and must exit cleanly. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`.

## Key takeaways

- Explained variance is the honest number. `components.csv` tells you what each axis means.
- Projection is a linear model: `transform` on new rows is a matrix product, nothing else.
- Use the 2-D embedding with a clustering algorithm to see groups that a 4-D fit misses.

## Next module

`07_tsne`: a non-linear embedding for visualisation when PCA gaps the groups.