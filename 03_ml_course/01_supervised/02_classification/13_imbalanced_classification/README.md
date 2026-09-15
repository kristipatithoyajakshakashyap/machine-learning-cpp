# Imbalanced Classification

## Problem and core idea

When one class is rare, accuracy is dominated by the majority class and a model that predicts 'negative' everywhere looks excellent. This module weights the log-loss so each positive row counts w times, which shifts the boundary toward recall on the rare class, and evaluates with macro F1, per-class recall and the precision-recall curve instead of accuracy. The positive weight is selected by cross-validation like any other hyper-parameter.

## Dataset and why

The course has no naturally imbalanced tabular set, so `ml::make_imbalanced(load_breast_cancer, 1.0, 30, 42)` keeps all 357 benign rows and a seeded subset of 30 malignant rows: 387 rows with 7.8% positives, original row ids preserved. Seed 42 makes the subset reproducible; the exercises change it. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_logistic_regression`, precision/recall/ROC from `10_model_evaluation`, and stratified splitting. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp / Model.cpp` | - | `course::WeightedLogistic`: full-batch gradient descent on a class-weighted log-loss, `predict_proba`, save/load | - |
| `tests/model_test.cpp` | `cimb_tests` | fixture test: unweighted fit separates two clusters, `predict_proba` rows sum to one, w = 8 never lowers positive recall versus w = 1, the weight survives save/load | prints only |
| `01_theory.cpp` | `cimb_theory` | Why accuracy misleads when one class is rare | prints only |
| `02_math_intuition.cpp` | `cimb_math_intuition` | The weighted log-loss gradient on one hand example | prints only |
| `03_implementation.cpp` | `cimb_implementation` | Sweep w in {1, 2, 4, 8} on one fixed split; recall against weight | `results/03_implementation_results/`: `weight_sweep.csv`, `recall_vs_weight.svg` |
| `04_end_to_end.cpp` | `cimb_end_to_end` | Full project on imbalanced breast cancer (387 rows, 7.8% positive): `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid positive weight w in {1, 2, 4, 8} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cimb_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; the `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_cimb (Model.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cimb_end_to_end cimb_predict cimb_tests
build\03_ml_course\01_supervised\02_classification\13_imbalanced_classification\cimb_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\13_imbalanced_classification\cimb_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\13_imbalanced_classification\cimb_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `cimb_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cimb_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R cimb
```

`cimb_numerical` runs `tests/model_test.cpp` (the fixture checks listed in the table; any failed check throws, so the exit code is non-zero). `cimb_workflow` runs `cimb_end_to_end --quick` and fails if the pipeline or its reload verification fails.

## Key takeaways

- Accuracy is high for every w; read macro F1 and `evaluation/per_class.csv` instead.
- Class weights move the threshold implicitly; `evaluation/pr_curve.csv` shows the whole trade-off.
- Stratify every split, or a fold may contain no positives at all.

## Next module

`../03_model_diagnostics/01_feature_importance_and_learning_curves`: interrogating a fitted model.
