# Gaussian Naive Bayes

## Problem and core idea

Gaussian naive Bayes assumes the features are conditionally independent given the class and each follows a class-specific normal distribution. Fitting is just per-class means, variances and priors; prediction applies Bayes' rule in log space to avoid underflow. The independence assumption is wrong for most data yet the classifier is often surprisingly good, fast and calibrated enough for a baseline. A variance smoothing term keeps near-constant features from producing infinite densities.

## Dataset and why

`iris` and `wine` (3 classes each) in the lesson so the per-class Gaussians can be inspected. The end-to-end project uses `breast_cancer` with variance smoothing in {1e-9, 1e-5, 0.01}. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

Bayes' rule, the normal density, logarithms, and `01_logistic_regression` for the discriminative counterpart. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `GaussianNB.hpp / GaussianNB.cpp` | - | `ml::GaussianNB`: class priors, per-feature means and variances, log-space posteriors, save/load | - |
| `01_theory.cpp` | `cnb_theory` | Gaussian naive Bayes in one page | prints only |
| `02_math_intuition.cpp` | `cnb_math_intuition` | Posteriors in log space, by hand | prints only |
| `03_implementation.cpp` | `cnb_implementation` | Naive Bayes on iris and wine with seeded splits | `results/03_implementation_results/`: `c04_nb_iris.split`, `c04_nb_wine.split` |
| `04_end_to_end.cpp` | `cnb_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid variance smoothing in {1e-9, 1e-5, 0.01} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cnb_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_cnb (GaussianNB.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cnb_end_to_end cnb_predict
build\03_ml_course\01_supervised\02_classification\04_naive_bayes\cnb_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\04_naive_bayes\cnb_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\04_naive_bayes\cnb_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `cnb_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cnb_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cnb` matches nothing. The smoke check is `cnb_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Fitting is a single pass over the data; there is nothing to tune except smoothing.
- Sum log-probabilities, never multiply probabilities: 30 features underflow a double.
- Correlated features are double-counted, which makes the posteriors over-confident.

## Next module

`05_decision_tree_classifier`: splits chosen by Gini impurity.
