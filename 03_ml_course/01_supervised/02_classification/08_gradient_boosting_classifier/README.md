# Gradient Boosting Classifier

## Problem and core idea

Gradient boosting for classification fits one small regression tree per class and per round to the gradient of the multinomial log-loss, then applies softmax to the accumulated scores. The learning rate scales each tree's contribution and controls how many rounds are needed. It usually matches or beats a forest on tabular data but has more hyper-parameters and overfits with too many rounds.

## Dataset and why

`titanic` in the lesson because its mixed, noisy features are where boosting's flexibility shows. The end-to-end project uses `breast_cancer` with learning rate in {0.03, 0.1, 0.2} at 40 rounds of depth-2 trees. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_regression/05_gradient_boosting_regressor`, softmax, and the log-loss gradient from `01_logistic_regression`. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `GradientBoostedTrees.hpp / GradientBoostedTrees.cpp` | - | `ml::GradientBoostedTrees` in classification mode: per-class residual trees, softmax scores, `predict_proba`, save/load | - |
| `01_theory.cpp` | `cgbc_theory` | Gradient boosted classification in one page | prints only |
| `02_math_intuition.cpp` | `cgbc_math_intuition` | One softmax round of boosting, by hand | prints only |
| `03_implementation.cpp` | `cgbc_implementation` | Boosting on titanic with a seeded split | `results/03_implementation_results/`: `c08_gbdt.split` |
| `04_end_to_end.cpp` | `cgbc_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid learning rate in {0.03, 0.1, 0.2} (40 rounds, depth 2) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cgbc_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_cgbc (GradientBoostedTrees.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cgbc_end_to_end cgbc_predict
build\03_ml_course\01_supervised\02_classification\08_gradient_boosting_classifier\cgbc_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\08_gradient_boosting_classifier\cgbc_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\08_gradient_boosting_classifier\cgbc_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `cgbc_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cgbc_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cgbc` matches nothing. The smoke check is `cgbc_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Each round fits K trees for K classes. The cost grows with the number of classes.
- Learning rate and rounds are coupled: halve one, roughly double the other.
- Boosted probabilities are usually better calibrated than a forest's vote fractions.

## Next module

`09_mlp_classifier`: a softmax head on learned features.