# Gaussian Process Regression

## Problem and core idea

A Gaussian process puts a prior over functions defined by a kernel, conditions on the training targets, and returns a posterior mean and variance for every query point. The posterior is exact linear algebra on the kernel matrix: one Cholesky factorisation gives the mean, the predictive standard deviation and the log marginal likelihood used to compare kernels. Cost is O(n^3), so the model shines on small data where calibrated uncertainty matters.

## Dataset and why

15 noisy sin(x) samples in the lesson so the posterior band can be drawn. The end-to-end project subsamples `diabetes` to 300 rows (the cubic cost) and tunes the RBF length scale over {0.3, 1, 3, 10}; afterwards it reloads the saved model and writes a 95% predictive band for every holdout row. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

Multivariate Gaussians (conditioning), the Cholesky factorisation, RBF kernels, and `08_svr` for the kernel idea. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp` | - | `course::GaussianProcess` (header-only): RBF kernel, Cholesky solve, `predict`, `predict_std`, `log_marginal_likelihood`, save/load | - |
| `tests/model_test.cpp` | `gp_tests` | fixture test: interpolation of noise-free data to 1e-4, finite marginal likelihood, small std at training points and large std far away, exact save/load round trip, rejection of invalid hyper-parameters and shapes | prints only |
| `01_theory.cpp` | `gp_theory` | Functions as random variables; what the kernel encodes | prints only |
| `02_math_intuition.cpp` | `gp_math_intuition` | Numbers behind the prior and the marginal likelihood | prints only |
| `03_implementation.cpp` | `gp_implementation` | Fit 15 noisy sine samples, plot the posterior band and the marginal-likelihood curve | `results/03_implementation_results/`: `posterior.csv`, `posterior.svg`, `marginal_likelihood.csv`, `marginal_likelihood.svg` |
| `04_end_to_end.cpp` | `gp_end_to_end` | Full project on diabetes (300-row subsample): `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid length scale in {0.3, 1, 3, 10}; then appends `evaluation/predictive_std.csv` | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `gp_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; the `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_core (Model.hpp is header-only)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target gp_end_to_end gp_predict gp_tests
build\03_ml_course\01_supervised\01_regression\09_gaussian_process\gp_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\09_gaussian_process\gp_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\09_gaussian_process\gp_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `gp_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by gp_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R gp
```

`gp_numerical` runs `tests/model_test.cpp` (the fixture checks listed in the table; any failed check throws, so the exit code is non-zero). `gp_workflow` runs `gp_end_to_end --quick` and fails if the pipeline or its reload verification fails.

## Key takeaways

- The GP posterior is closed form: no optimiser, only a Cholesky solve.
- Predictive std grows away from the data, which no other model in this group provides.
- The marginal likelihood is a principled way to pick kernel hyper-parameters without a validation set.

## Next module

`../02_classification/01_logistic_regression`: the same linear machinery for a binary target.
