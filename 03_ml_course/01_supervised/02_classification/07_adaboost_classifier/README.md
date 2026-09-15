# AdaBoost Classifier

## Problem and core idea

AdaBoost trains a sequence of weak learners (depth-1 or shallow trees) where each round reweights the rows so that the previously misclassified ones matter more, and combines the learners with weights derived from their error. The SAMME variant extends it to several classes. The final model is an additive vote, so it is a boosting method like gradient boosting but with an exponential loss and a closed-form learner weight.

## Dataset and why

`breast_cancer` in both the lesson and the end-to-end project (rounds in {20, 50, 100} with learning rate 0.5). Weak stumps on 30 features show the reweighting dynamics clearly. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`05_decision_tree_classifier`, weighted training sets, and the exponential loss at an intuitive level. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `AdaBoost.hpp / AdaBoost.cpp` | - | `ml::AdaBoost` (SAMME): weighted stump training, learner weights, `predict_proba`, save/load | - |
| `01_theory.cpp` | `cada_theory` | AdaBoost in one page | prints only |
| `02_math_intuition.cpp` | `cada_math_intuition` | The weight update table, by hand | prints only |
| `03_implementation.cpp` | `cada_implementation` | AdaBoost on breast cancer with a seeded split | `results/03_implementation_results/`: `c09_adaboost.split` |
| `04_end_to_end.cpp` | `cada_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid rounds in {20, 50, 100} (learning rate 0.5) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cada_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_cada (AdaBoost.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cada_end_to_end cada_predict
build\03_ml_course\01_supervised\02_classification\07_adaboost_classifier\cada_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\07_adaboost_classifier\cada_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\07_adaboost_classifier\cada_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `cada_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cada_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cada` matches nothing. The smoke check is `cada_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- A learner with error above 1/2 gets a negative weight in the binary case; SAMME shifts the threshold for K classes.
- Row weights concentrate on hard examples, which makes AdaBoost sensitive to label noise.
- Stumps are enough: the strength comes from the sequence, not from any one learner.

## Next module

`08_gradient_boosting_classifier`: boosting with gradients of the log-loss.
