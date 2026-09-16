# Feature Importance and Learning Curves

## Problem and core idea

Two questions about any fitted model: which inputs does it rely on, and would more data help? Permutation importance answers the first by shuffling one column at a time and measuring how much a metric drops. A learning curve answers the second by refitting on growing fractions of the training rows and plotting train and cross-validated scores. Both are model-agnostic: they need only `predict` and a metric.

## Dataset and why

`wine` (178 rows, 13 features, 3 classes) with the course random forest. The chemistry features are redundant, so several real columns get near-zero importance, and the small size makes the learning curve's right edge informative. Lesson 4 appends a `pure_noise` column that must rank last. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

Random forests (`01_regression/04`, `02_classification/06`), k-fold cross-validation (`02_classification/10`), and C++ lambdas/templates. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Forest.hpp` | - | `course::make_forest(n_trees, depth, min_leaf)`: builds `ml::RandomForest` in classification mode (header-only) | - |
| `helper/eval/diagnostics.hpp` | - | `ml::permutation_importance` and `ml::learning_curve`, shared with later modules | - |
| `tests/model_test.cpp` | `diag_tests` | fixture test: one importance row per column, shuffling the only informative column costs a lot, a constant column costs nothing, learning-curve `n_train` grows with the fraction and the CV score at 100% is not below that at 10% | prints only |
| `01_theory.cpp` | `diag_theory` | Concepts plus a toy forest | prints only |
| `02_math_intuition.cpp` | `diag_math_intuition` | Importance formula on a two-feature fixture | prints only |
| `03_implementation.cpp` | `diag_implementation` | Both helpers on synthetic 4-feature data | `results/03_implementation_results/`: `importance.csv/.svg`, `learning_curve.csv/.svg` |
| `04_permutation_importance.cpp` | `diag_04_permutation_importance` | Out-of-fold importance on wine plus a noise column. It exits 1 if noise is not last | `results/04_permutation_importance_results/`: `permutation_importance.csv`, `permutation_importance.svg` |
| `05_learning_curve.cpp` | `diag_05_learning_curve` | 5-fold learning curve, fractions 0.1 to 1.0 | `results/05_learning_curve_results/`: `learning_curve.csv`, `learning_curve.svg` |
| `06_end_to_end.cpp` | `diag_end_to_end` | Full project on wine: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid n_estimators in {20, 50, 100}. Then it appends `evaluation/permutation_importance.csv` and `evaluation/figures/learning_curve.svg` from training rows | `results/06_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `diag_predict` | `#include`s `06_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. The `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_tree_models (Forest.hpp is header-only)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target diag_end_to_end diag_predict diag_tests
build\03_ml_course\01_supervised\03_model_diagnostics\01_feature_importance_and_learning_curves\diag_end_to_end            # full dataset -> results/06_end_to_end_results/full/
build\03_ml_course\01_supervised\03_model_diagnostics\01_feature_importance_and_learning_curves\diag_end_to_end --quick    # seeded 400-row subset -> results/06_end_to_end_results/quick/
build\03_ml_course\01_supervised\03_model_diagnostics\01_feature_importance_and_learning_curves\diag_predict --predict results\06_end_to_end_results\full\data\holdout_features.csv --model results\06_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `diag_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/06_end_to_end_results/
  full/   or   quick/
    data/          row_assignments.csv, holdout_features.csv, holdout.split
    eda/           summaries and figures computed on training rows only
    validation/    candidate_scores.csv, summary.csv, figures/parameter_curve.svg
    evaluation/    metrics.json, predictions.csv, per-class or residual files, figures/
    model/         model_state.txt, preprocessing_state.txt, target_state.txt, features.txt
    inference/     reload_verification.json
    run_manifest.json, report.md, execution.log
results/predict_results/predictions.csv        (written by diag_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R diag
```

`diag_numerical` runs `tests/model_test.cpp`. The fixture checks are listed in the table. Any failed check throws, so the exit code is non-zero. `diag_workflow` runs `diag_end_to_end --quick`. It fails if the pipeline or its reload verification fails.

## Key takeaways

- Importance well above its std matters. Near-zero means the model works without that column.
- Measure importance out of fold when possible: in-sample importance rewards memorised noise.
- A flat CV curve at the right edge means more data will not help. A rising one means it will.

## Next module

`02_feature_selection`: acting on importance by dropping columns.