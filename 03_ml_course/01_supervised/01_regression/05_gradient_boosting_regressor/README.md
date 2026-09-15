# Gradient Boosting Regressor

## Problem and core idea

Gradient boosting builds an additive model one small tree at a time. Each new tree is fitted to the negative gradient of the loss (for squared error, the current residuals), and its contribution is scaled by a learning rate. Small trees plus many rounds give a low-bias model whose variance is controlled by the learning rate and the number of rounds. Unlike a forest, the trees are dependent and order matters.

## Dataset and why

`diabetes` (442 x 10) in the lesson so the residual table can be printed round by round. The end-to-end project uses California housing with a learning-rate grid {0.03, 0.1, 0.2} at 50 rounds of depth-3 trees. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`03_decision_tree_regressor`, gradients of the squared-error loss, and the idea of an additive model. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `GradientBoostedTrees.hpp / GradientBoostedTrees.cpp` | - | `ml::GradientBoostedTrees`: residual fitting with shrinkage, regression and softmax classification via `set_task`, save/load | - |
| `01_theory.cpp` | `rgbr_theory` | Gradient boosted trees in one page | prints only |
| `02_math_intuition.cpp` | `rgbr_math_intuition` | The residual table, by hand | prints only |
| `03_implementation.cpp` | `rgbr_implementation` | Boosting on diabetes with a seeded split | `results/03_implementation_results/`: `s09_gbdt.split` |
| `04_end_to_end.cpp` | `rgbr_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid learning rate in {0.03, 0.1, 0.2} (50 rounds, depth 3) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `rgbr_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_rgbr (GradientBoostedTrees.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target rgbr_end_to_end rgbr_predict
build\03_ml_course\01_supervised\01_regression\05_gradient_boosting_regressor\rgbr_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\05_gradient_boosting_regressor\rgbr_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\05_gradient_boosting_regressor\rgbr_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `rgbr_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by rgbr_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R rgbr` matches nothing. The smoke check is `rgbr_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- A smaller learning rate needs more rounds but generalises better: the two are traded against each other.
- Boosting reduces bias; bagging reduces variance. Choose by what your single tree suffers from.
- Too many rounds overfit, so the number of rounds is a hyper-parameter, unlike a forest's tree count.

## Next module

`06_mlp_regressor`: a model whose features are learned rather than split.
