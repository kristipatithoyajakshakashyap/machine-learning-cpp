# Decision Tree Regressor

## Problem and core idea

A CART regression tree splits the feature space into axis-aligned boxes and predicts the mean target inside each box. Each split is chosen greedily to reduce the variance of the target the most. Depth controls capacity: a deep tree memorises the training set, a shallow one underfits. Trees need no scaling and handle interactions natively, which is why they are the building block of forests and boosting.

## Dataset and why

`diabetes` (442 x 10) for the implementation lesson because it fits in a second and exposes overfitting clearly. The depth curve and the end-to-end project use California housing (subsampled to 4,000 rows in the curve lesson for speed), where the depth grid {3, 6, 10} produces a visible bias-variance sweet spot. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

Variance, greedy search, recursion, and the train/test discipline from `01_linear_regression`. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `DecisionTree.hpp / DecisionTree.cpp` | - | `ml::DecisionTree`: CART splits for regression and classification, `max_depth` and `min_leaf` stopping rules, save/load (shared internals in `helper/models`) | - |
| `01_theory.cpp` | `rdtr_theory` | CART regression trees in one page | prints only |
| `02_math_intuition.cpp` | `rdtr_math_intuition` | Variance reduction of one split, by hand | prints only |
| `03_implementation.cpp` | `rdtr_implementation` | Tree on diabetes with a seeded split | `results/03_implementation_results/`: `s07_tree.split` |
| `04_depth_curve.cpp` | `rdtr_04_depth_curve` | Train/test R2 versus `max_depth` on 4,000 California rows | `results/04_depth_curve_results/`: `depth_curve.svg` |
| `05_end_to_end.cpp` | `rdtr_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid max_depth in {3, 6, 10} (min_leaf 5) | `results/05_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `rdtr_predict` | `#include`s `05_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_rdtr (DecisionTree.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target rdtr_end_to_end rdtr_predict
build\03_ml_course\01_supervised\01_regression\03_decision_tree_regressor\rdtr_end_to_end            # full dataset -> results/05_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\03_decision_tree_regressor\rdtr_end_to_end --quick    # seeded 400-row subset -> results/05_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\03_decision_tree_regressor\rdtr_predict --predict results\05_end_to_end_results\full\data\holdout_features.csv --model results\05_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `rdtr_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by rdtr_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R rdtr` matches nothing. The smoke check is `rdtr_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Greedy variance reduction is local. A split that looks poor now might enable a great one later, and CART never sees it.
- Train R2 goes to 1 with depth. Only the test curve tells you where to stop.
- One tree is high-variance. Averaging many (next module) is the fix.

## Next module

`04_random_forest_regressor`: bootstrap many trees and average them.