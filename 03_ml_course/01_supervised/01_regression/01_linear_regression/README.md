# Linear Regression (OLS, Ridge, Lasso, Polynomial)

## Problem and core idea

Predict a continuous target as a weighted sum of the input features. Ordinary least squares picks the weights that minimise the squared error, and the normal equations give them in closed form. Ridge adds an L2 penalty that shrinks the slopes; lasso adds an L1 penalty that drives some of them to exactly zero, so it also selects features. Polynomial features show that 'linear' means linear in the weights, not in the inputs.

## Dataset and why

Small lessons use `diabetes` (442 rows, 10 standardised features) because OLS on it matches scikit-learn to many digits, and `tips` (244 rows) for one-feature plots. The end-to-end project uses California housing (20,640 rows, 8 features) so the ridge penalty grid actually matters and the holdout is large enough for tight bootstrap intervals. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

Matrix algebra (solving a linear system), mean/variance, an 80/20 train/test split, and C++ `std::vector`. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `LinearRegression.hpp / LinearRegression.cpp` | - | `ml::LinearRegression`: normal equations, ridge, coordinate-descent lasso, polynomial features, save/load | - |
| `01_theory.cpp` | `lin_theory` | Linear model, OLS, ridge and lasso on three toy points | prints only |
| `02_math_intuition.cpp` | `lin_math_intuition` | Gradient descent versus the closed form on tips | prints only |
| `03_implementation.cpp` | `lin_implementation` | OLS on diabetes with a seeded 80/20 split | `results/03_implementation_results/`: `s01_ols.split` |
| `04_ridge_regression.cpp` | `lin_04_ridge_regression` | Ridge penalty sweep on the same diabetes split | `results/04_ridge_regression_results/`: `s01_ols.split` |
| `05_lasso_regression.cpp` | `lin_05_lasso_regression` | Lasso coefficient path as lambda grows | `results/05_lasso_regression_results/`: `coefficient_path.csv`, `coefficient_path.svg` |
| `07_polynomial_regression.cpp` | `lin_07_polynomial_regression` | Degree sweep of polynomial features on tips | `results/07_polynomial_regression_results/`: `s05_poly.split` |
| `08_end_to_end.cpp` | `lin_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid ridge lambda in {0.1, 1, 10, 100} | `results/08_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `lin_predict` | `#include`s `08_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_rlin (LinearRegression.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target lin_end_to_end lin_predict
build\03_ml_course\01_supervised\01_regression\01_linear_regression\lin_end_to_end            # full dataset -> results/08_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\01_linear_regression\lin_end_to_end --quick    # seeded 400-row subset -> results/08_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\01_linear_regression\lin_predict --predict results\08_end_to_end_results\full\data\holdout_features.csv --model results\08_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `lin_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/08_end_to_end_results/
  full/   or   quick/
    data/          row_assignments.csv, holdout_features.csv, holdout.split
    eda/           summaries and figures computed on training rows only
    validation/    candidate_scores.csv, summary.csv, figures/parameter_curve.svg
    evaluation/    metrics.json, predictions.csv, per-class or residual files, figures/
    model/         model_state.txt, preprocessing_state.txt, target_state.txt, features.txt
    inference/     reload_verification.json
    run_manifest.json, report.md, execution.log
results/predict_results/predictions.csv        (written by lin_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R lin` matches nothing. The smoke check is `lin_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- The normal equations solve OLS exactly; iterative solvers only matter when the design is huge.
- Ridge shrinks every slope, lasso zeroes some: use lasso when you want a sparse model.
- Never penalise the intercept, and standardise features before penalising anything.
- Polynomial features let a linear solver fit curves, at the price of variance as the degree grows.

## Next module

`02_knn_regression`: a model with no weights at all, only distances.
