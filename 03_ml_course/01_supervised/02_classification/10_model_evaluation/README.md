# Model Evaluation

## Problem and core idea

A score on the training set says nothing about generalisation, and a single test score says little about its uncertainty. This module builds every classification metric (confusion matrix, precision, recall, F1, ROC and PR curves, calibration) from one list of predictions, then shows the discipline the whole course uses: hold out first, cross-validate on training rows only, choose the model family by CV, and touch the holdout once. The end-to-end project applies that discipline to choosing among four regression families.

## Dataset and why

`breast_cancer` for the binary deep dive and `titanic` for the head-to-head shoot-out of every classifier in this group. The end-to-end project uses California housing and treats the model family (0 ridge, 1 kNN, 2 forest, 3 MLP) as the 'parameter' the pipeline selects by training CV. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

Every earlier module of `02_classification` (their models are all linked here) and k-fold cross-validation. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `(no model source)` | - | links the models of `01_regression` and `02_classification`; the end-to-end `ComparisonModel` wraps four of them behind one save/load interface | - |
| `01_theory.cpp` | `eval_theory` | Binary-classification deep dive on breast cancer: every metric explained | `results/01_theory_results/`: `c12_binary.split` |
| `02_math_intuition.cpp` | `eval_math_intuition` | Every metric from one list of predictions, by hand | prints only |
| `03_implementation.cpp` | `eval_implementation` | Every classifier of this group head-to-head on titanic with k-fold CV | `results/03_implementation_results/`: `c13_shootout*.split` |
| `04_end_to_end.cpp` | `eval_end_to_end` | Full project on California housing (regression): `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid model family in {0 ridge, 1 kNN, 2 forest, 3 MLP} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `eval_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_eval` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target eval_end_to_end eval_predict
build\03_ml_course\01_supervised\02_classification\10_model_evaluation\eval_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\10_model_evaluation\eval_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\10_model_evaluation\eval_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `eval_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by eval_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R eval` matches nothing. The smoke check is `eval_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Accuracy hides class imbalance; always look at per-class recall and the PR curve too.
- Selecting a model on the holdout and then reporting the holdout score is optimistic: select by CV.
- Bootstrap intervals in `evaluation/metrics.json` tell you whether two models are really different.

## Next module

`11_lda`: a generative linear classifier.
