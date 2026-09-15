# k-Nearest Neighbours Classification

## Problem and core idea

Classify a query by majority vote among its k nearest training rows; class probabilities are the vote fractions. There is no training, the decision boundary can be arbitrarily complex, and k controls smoothness. Ties are broken deterministically so results are reproducible. Distance scaling matters as much as in the regression version.

## Dataset and why

`iris` (150 x 4, 3 classes) in the lesson because its clusters make the neighbour idea obvious. The end-to-end project uses `breast_cancer` with k in {3, 7, 15} so every classification module is compared on the same holdout. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_regression/02_knn_regression` (this module forwards to the same `ml::KNN` internals) and basic voting. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `KNN.hpp / KNN.cpp` | - | `ml::KNNClassifier`: vote counting with deterministic tie-break, `predict_proba`, save/load | - |
| `01_theory.cpp` | `cknn_theory` | k-nearest-neighbour classification in one page | prints only |
| `02_math_intuition.cpp` | `cknn_math_intuition` | Voting and the tie-break, by hand | prints only |
| `03_implementation.cpp` | `cknn_implementation` | kNN on iris with a sweep over k | `results/03_implementation_results/`: `c02_knn.split` |
| `04_end_to_end.cpp` | `cknn_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid k in {3, 7, 15} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cknn_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_cknn (KNN.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cknn_end_to_end cknn_predict
build\03_ml_course\01_supervised\02_classification\02_knn_classification\cknn_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\02_knn_classification\cknn_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\02_knn_classification\cknn_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `cknn_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cknn_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cknn` matches nothing. The smoke check is `cknn_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Odd k avoids most ties in binary problems; the code still defines what happens on a tie.
- Vote fractions are coarse probabilities (multiples of 1/k), which shows in the calibration plot.
- kNN is a strong baseline that any learned model should beat on the same split.

## Next module

`03_svm_classification`: a linear boundary chosen by margin.
