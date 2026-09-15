# Random Forest Regressor

## Problem and core idea

A random forest averages many decision trees, each grown on a bootstrap sample of the rows and restricted to a random subset of features at every split. Averaging reduces the variance of a single deep tree without adding bias, and the feature subsampling decorrelates the trees so the averaging actually helps. The number of trees only needs to be 'large enough'; depth and leaf size still control each tree's capacity.

## Dataset and why

California housing throughout (subsampled to 4,000 rows in the lessons for speed, full 20,640 rows in the end-to-end project). It has enough rows and enough non-linear structure for a forest to beat the linear baseline clearly. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`03_decision_tree_regressor`, bootstrap sampling, averaging independent estimates reduces variance. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `RandomForest.hpp / RandomForest.cpp` | - | `ml::RandomForest`: bagged trees with feature subsampling for regression or classification, `set_task`, save/load (used again by `03_model_diagnostics`) | - |
| `01_theory.cpp` | `rrf_theory` | Random forest regression in one page | prints only |
| `02_math_intuition.cpp` | `rrf_math_intuition` | What a bootstrap sample contains (about 63% unique rows) | prints only |
| `03_implementation.cpp` | `rrf_implementation` | Forest on a California subsample versus a single tree | `results/03_implementation_results/`: `s08_forest.split` |
| `04_n_estimators.cpp` | `rrf_04_n_estimators` | Test R2 versus number of trees | `results/04_n_estimators_results/`: `n_estimators.svg` |
| `05_end_to_end.cpp` | `rrf_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid max_depth in {4, 8, 12} with 40 trees | `results/05_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `rrf_predict` | `#include`s `05_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_rrf (RandomForest.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target rrf_end_to_end rrf_predict
build\03_ml_course\01_supervised\01_regression\04_random_forest_regressor\rrf_end_to_end            # full dataset -> results/05_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\04_random_forest_regressor\rrf_end_to_end --quick    # seeded 400-row subset -> results/05_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\04_random_forest_regressor\rrf_predict --predict results\05_end_to_end_results\full\data\holdout_features.csv --model results\05_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `rrf_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/05_end_to_end_results/
  full/   or   quick/
    data/          row_assignments.csv, holdout_features.csv, holdout.split
    eda/           summaries and figures computed on training rows only
    validation/    candidate_scores.csv, summary.csv, figures/parameter_curve.svg
    evaluation/    metrics.json, predictions.csv, per-class or residual files, figures/
    model/         model_state.txt, preprocessing_state.txt, target_state.txt, features.txt
    inference/     reload_verification.json
    run_manifest.json, report.md, execution.log
results/predict_results/predictions.csv        (written by rrf_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R rrf` matches nothing. The smoke check is `rrf_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- More trees never hurt accuracy, they only cost time; the curve flattens quickly.
- Feature subsampling is what makes averaging work: identical trees average to the same tree.
- Forests are hard to overfit with `n_estimators` but still overfit through depth.

## Next module

`05_gradient_boosting_regressor`: fit trees sequentially to residuals instead of in parallel.
