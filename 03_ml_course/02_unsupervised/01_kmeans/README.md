# K-means

## Problem and core idea

Partition rows into `k` groups by minimizing the sum of squared distances to their assigned centroid, the inertia. The implementation starts from k-means++ seeds and runs Lloyd iterations with `n_init` restarts. `predict` assigns the nearest centroid. Predictions are closed-form and cheap: given a trained model, assigning a new row is only Euclidean distance to `k` centroids.

The module preprocesses `X` with the shared `Preprocessor` (median imputation then z-score from `helper/preprocessing/preprocessor.hpp`) before fitting, because k-means is not scale-invariant.

## Dataset and why

Iris (`helper/data/iris.csv`, 150 rows, 4 numeric columns, 3 known species) for the implementation lesson and Palmer Penguins (`helper/data/penguins.csv`, 344 rows, 4 numeric columns, 3 species) for the end-to-end project. Both have compact, well-separated groups, so the `k` grid {2, 3, 4, 5, 6} produces a clear elbow in the inertia curve and a clear silhouette peak. Penguins also mix units (millimetres versus grams), so standardization visibly matters. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

C++17 (vectors, lambdas, `std::function`), Euclidean distance, and means/variance. Read `theory.md`, `math_intuition.md` and `implementation.md`, then run the numbered lessons in order.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `KMeans.hpp / KMeans.cpp` | library `ml_kmeans` | `ml::KMeans`: k-means++ seeding, Lloyd iterations, `n_init` restarts, `predict` (nearest centroid), save/load | - |
| `01_theory.cpp` | `ukm_theory` | Partition, inertia, why k-means gets stuck | prints only |
| `02_math_intuition.cpp` | `ukm_math_intuition` | Two centroids on four points, by hand | prints only |
| `03_implementation.cpp` | `ukm_implementation` | Iris, k=3, seed 42: cluster sizes, species best-match accuracy, inertia, iterations, silhouette | `results/03_implementation_results/`: `iris_kmeans.svg`, `iris_kmeans3.labels` |
| `04_end_to_end.cpp` | `ukm_end_to_end` | Full project on penguins: shared `workflow.hpp` helpers (manifest, report, stability, snapshot, verify): impute with medians, standardise, grid `k` in {2, 3, 4, 5, 6} scored by silhouette, final fit on all rows, save model + preprocessing + `feature_schema.csv`, reload check | `results/04_end_to_end_results/` (see below) |
| `predict.cpp` | `ukm_predict` | Takes one argument, a header-less numeric CSV whose columns match `feature_schema.csv`. It restores preprocessing and model from `results/04_end_to_end_results/model/` and emits `row_id` and the cluster id | `results/predict_results/new_predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_kmeans (KMeans.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target ukm_end_to_end ukm_predict
build\03_ml_course\02_unsupervised\01_kmeans\ukm_end_to_end            # full run
build\03_ml_course\02_unsupervised\01_kmeans\ukm_predict <numeric_csv>
```

The end-to-end program takes no arguments, because a full run on 344 rows takes seconds. `ukm_predict <numeric_csv>` expects its single argument to be a header-less CSV with the four features in the order of `results/04_end_to_end_results/model/feature_schema.csv`, in raw units. Blank, `NA` or `nan` fields are imputed with the saved training medians. Output column `output_0` is the cluster id. `03_ml_course/02_unsupervised/tests/penguin_inference.csv` is a ready-made example. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module. The project run writes:

```
results/04_end_to_end_results/
  data/          row sources and preprocessing notes
  eda/           per-feature summaries and figures
  model/         model_state.txt, preprocessing_state.txt, feature_schema.csv
  validation/    candidate_scores.csv, selected_parameters.json, stability.csv
  evaluation/    metrics.json, assignments.csv, cluster_profiles.csv, figures/
  inference/     reload_verification.json
  run_manifest.json, report.md, execution.log
results/predict_results/new_predictions.csv        (written by ukm_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R ukm
```

`ukm_workflow` runs `ukm_end_to_end` and fails if any stage or the reload verification fails. `ukm_fresh_reload` starts a new process, loads the saved model and preprocessor, recomputes the labels and matches them against `evaluation/assignments.csv`. `ukm_new_rows` runs `ukm_predict` on `tests/penguin_inference.csv` and must exit cleanly. The unit checks in `../tests/test_unsupervised.cpp` run under `unsupervised_numerical`.

## Key takeaways

- k-means minimizes inertia. Silhouette, not inertia, picks `k`, because inertia always falls as `k` grows.
- k-means++ seeds beat random init and make `n_init` less necessary.
- `predict` over new rows assigns the nearest centroid with the same preprocessing.

## Next module

`02_agglomerative`: a hierarchy instead of a flat partition.