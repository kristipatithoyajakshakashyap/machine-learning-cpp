# Logistic Regression

## Problem and core idea

Logistic regression models the probability of the positive class as a sigmoid of a linear score and fits the weights by minimising the log-loss with gradient descent. The decision boundary is linear, the probabilities are calibrated if the model is right, and an L2 penalty controls overfitting. The one-vs-rest and softmax (multinomial) extensions handle more than two classes.

## Dataset and why

`breast_cancer` (569 rows, 30 features, 2 classes) is the binary workhorse of this group: linearly separable enough for a strong baseline, with enough rows for meaningful ROC and calibration plots. Lesson 4 uses `wine` (178 x 13, 3 classes) for the multiclass variants. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_regression/01_linear_regression`, the sigmoid function, and gradient descent. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `LogisticRegression.hpp / LogisticRegression.cpp` | - | `ml::LogisticRegression`: binary and multinomial, L2 penalty, gradient descent, `predict_proba`, save/load | - |
| `01_theory.cpp` | `clog_theory` | Logistic regression in one page | prints only |
| `02_math_intuition.cpp` | `clog_math_intuition` | The sigmoid and one gradient step, by hand | prints only |
| `03_implementation.cpp` | `clog_implementation` | Binary model on breast cancer with a seeded split | `results/03_implementation_results/`: `c01_logistic.split` |
| `04_multiclass_and_softmax.cpp` | `clog_04_multiclass_and_softmax` | One-vs-rest versus softmax on wine | `results/04_multiclass_and_softmax_results/`: `c11_multiclass.split` |
| `05_end_to_end.cpp` | `clog_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid L2 penalty in {0.1, 1, 10, 100} (lr 0.03, 600 epochs) | `results/05_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `clog_predict` | `#include`s `05_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_clog (LogisticRegression.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target clog_end_to_end clog_predict
build\03_ml_course\01_supervised\02_classification\01_logistic_regression\clog_end_to_end            # full dataset -> results/05_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\01_logistic_regression\clog_end_to_end --quick    # seeded 400-row subset -> results/05_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\01_logistic_regression\clog_predict --predict results\05_end_to_end_results\full\data\holdout_features.csv --model results\05_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `clog_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by clog_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R clog` matches nothing. The smoke check is `clog_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Log-loss is convex, so gradient descent finds the global optimum; the learning rate only affects speed.
- Probabilities, not just labels, are the output: read `evaluation/figures/calibration.svg`.
- Softmax shares one normaliser across classes; one-vs-rest trains independent binary models.

## Next module

`02_knn_classification`: majority vote among neighbours.
