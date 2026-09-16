# k-Nearest Neighbours Regression

## Problem and core idea

Predict a target by averaging the targets of the k training rows closest to the query. There is no training step and no parametric form. The training set is the model. The single hyper-parameter k trades variance (small k memorises noise) against bias (large k smooths away structure). Because distances are computed in feature space, feature scaling decides which columns dominate.

## Dataset and why

`diabetes` (442 x 10) for the lesson because every feature is already standardised, so distances are meaningful without preprocessing. The end-to-end project uses California housing, where the pipeline's own standardisation is required and the k grid {3, 7, 15} shows the bias-variance trade-off. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

Euclidean distance, sorting, the 80/20 split idea, and `01_linear_regression` for the baseline it is compared against. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `KNN.hpp / KNN.cpp` | - | `ml::KNNRegressor` (and the classifier reused by `02_classification/02_knn_classification`): stores rows, brute-force neighbour search, save/load | - |
| `01_theory.cpp` | `rknn_theory` | What kNN assumes and why k is the whole model | prints only |
| `02_math_intuition.cpp` | `rknn_math_intuition` | Distances and the k-average worked by hand | prints only |
| `03_implementation.cpp` | `rknn_implementation` | kNN on diabetes with a sweep over k | `results/03_implementation_results/`: `s06_knn.split` |
| `04_end_to_end.cpp` | `rknn_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid k in {3, 7, 15} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `rknn_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_rknn (KNN.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target rknn_end_to_end rknn_predict
build\03_ml_course\01_supervised\01_regression\02_knn_regression\rknn_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\02_knn_regression\rknn_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\02_knn_regression\rknn_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `rknn_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/04_end_to_end_results/
  full/   or   quick/
    data/          row_assignments.csv, holdout_features.csv, holdout.split
    eda/           summaries and figures computed on training rows only
    validation/    candidate_scores.csv, summary.csv, figures/parameter_curve.svg
    evaluation/    metrics.json, predictions.csv, per-class or residual files, figures/
    model/         model_state.txt, preprocessing_state.txt, target_state.txt, features.txt
    inference/     reload_verification.json
    run_manifest.json, report.md, execution.log
results/predict_results/predictions.csv        (written by rknn_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R rknn` matches nothing. The smoke check is `rknn_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- k = 1 gives train R2 of 1 and the worst test R2. Watch the gap close as k grows.
- Prediction cost is O(n * p) per query: fast to fit, slow to serve.
- Standardise features or the largest-scale column silently becomes the only feature.

## Next module

`03_decision_tree_regressor`: piecewise-constant predictions chosen by variance reduction.