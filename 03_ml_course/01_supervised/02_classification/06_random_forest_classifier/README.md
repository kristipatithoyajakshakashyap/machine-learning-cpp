# Random Forest Classifier

## Problem and core idea

A random forest classifier grows many trees on bootstrap samples with random feature subsets at each split and predicts by majority vote. The vote fractions serve as probabilities. Bagging removes most of the variance of a single tree, and feature subsampling keeps the trees from all choosing the same dominant feature. It is the strongest default model in this group and needs almost no tuning.

## Dataset and why

`penguins` in the lesson (compare with the single tree of the previous module). The estimator curve and the end-to-end project use `breast_cancer` with max_depth in {3, 6, 10} and 40 trees. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`05_decision_tree_classifier`, `01_regression/04_random_forest_regressor`, and the bootstrap. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `RandomForest.hpp / RandomForest.cpp` | - | `ml::RandomForest` in classification mode: bagging, feature subsampling, vote-fraction `predict_proba`, save/load | - |
| `01_theory.cpp` | `crfc_theory` | Random forest classification in one page | prints only |
| `02_math_intuition.cpp` | `crfc_math_intuition` | Why feature subsampling decorrelates trees | prints only |
| `03_implementation.cpp` | `crfc_implementation` | Forest versus tree on penguins | `results/03_implementation_results/`: `c07_forest.split` |
| `04_estimators_curve.cpp` | `crfc_04_estimators_curve` | Accuracy versus number of trees on breast cancer | `results/04_estimators_curve_results/`: `n_estimators.svg` |
| `05_end_to_end.cpp` | `crfc_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid max_depth in {3, 6, 10} (40 trees, min_leaf 2) | `results/05_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `crfc_predict` | `#include`s `05_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_crfc (RandomForest.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target crfc_end_to_end crfc_predict
build\03_ml_course\01_supervised\02_classification\06_random_forest_classifier\crfc_end_to_end            # full dataset -> results/05_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\06_random_forest_classifier\crfc_end_to_end --quick    # seeded 400-row subset -> results/05_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\06_random_forest_classifier\crfc_predict --predict results\05_end_to_end_results\full\data\holdout_features.csv --model results\05_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `crfc_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by crfc_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R crfc` matches nothing. The smoke check is `crfc_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Accuracy saturates after a few dozen trees. More trees only tighten the probability estimates.
- Depth is the real capacity knob. The tree count is a cost knob.
- Vote fractions are better calibrated than a single tree's leaf fractions but still not perfect.

## Next module

`07_adaboost_classifier`: reweight the rows instead of resampling them.