# Decision Tree Classifier

## Problem and core idea

A classification tree partitions the feature space with axis-aligned splits chosen to reduce Gini impurity the most, and predicts the majority class of each leaf. Trees handle mixed scales and interactions without preprocessing and are easy to read, but a single deep tree has high variance. Depth and minimum leaf size are the capacity controls.

## Dataset and why

`penguins` (342 complete cases, 3 species) in the lesson because two or three splits already separate the species and the tree is short enough to print in full. The end-to-end project uses `breast_cancer` with max_depth in {2, 4, 8}. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_regression/03_decision_tree_regressor` (same CART machinery) and the Gini impurity formula. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `DecisionTree.hpp / DecisionTree.cpp` | - | `ml::DecisionTree` in classification mode: Gini splits, majority leaves, `predict_proba` from leaf class fractions, save/load | - |
| `01_theory.cpp` | `cdtc_theory` | Decision tree classification in one page | prints only |
| `02_math_intuition.cpp` | `cdtc_math_intuition` | Choosing a split by Gini reduction, by hand | prints only |
| `03_implementation.cpp` | `cdtc_implementation` | Tree on penguins with a seeded split | `results/03_implementation_results/`: `c03_tree.split` |
| `04_end_to_end.cpp` | `cdtc_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid max_depth in {2, 4, 8} (min_leaf 3) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cdtc_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_cdtc (DecisionTree.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cdtc_end_to_end cdtc_predict
build\03_ml_course\01_supervised\02_classification\05_decision_tree_classifier\cdtc_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\05_decision_tree_classifier\cdtc_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\05_decision_tree_classifier\cdtc_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `cdtc_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cdtc_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cdtc` matches nothing. The smoke check is `cdtc_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Gini and entropy pick almost the same splits. The stopping rule matters more than the criterion.
- A depth-2 tree on breast cancer is already competitive, which is a hint that the problem is easy.
- Leaf class fractions are the tree's probabilities and are badly calibrated for deep trees.

## Next module

`06_random_forest_classifier`: bagging the tree.