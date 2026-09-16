# Epsilon Support Vector Regression

## Problem and core idea

Epsilon-SVR fits a function that stays within a tube of half-width epsilon around the targets and pays a linear penalty, scaled by C, only for points outside it. Only those points (the support vectors) carry non-zero dual coefficients. With a kernel the same dual problem fits non-linear curves. The module solves the dual with FISTA plus a bisection that enforces the zero-sum constraint, and recovers the bias from the margin points.

## Dataset and why

A 60-point noisy sine in the lesson so the tube is visible in the SVG. The end-to-end project uses `diabetes` with an RBF kernel and C in {1, 10, 100}. The standardised features make one kernel width reasonable for every column. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

Linear regression, the hinge/epsilon-insensitive loss, Lagrangian duality at an intuitive level, and RBF kernels. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp / Model.cpp` | - | `course::SVR`: linear or RBF kernel, dual solver, `C`, `epsilon`, save/load | - |
| `tests/model_test.cpp` | `svr_tests` | fixture test: predictions within 0.1 of an exact line, dual coefficients bounded by C and summing to zero, save/load round trip, rejection of bad input | prints only |
| `01_theory.cpp` | `svr_theory` | The epsilon tube and the role of C | prints only |
| `02_math_intuition.cpp` | `svr_math_intuition` | Dual variables on a tiny problem, by hand | prints only |
| `03_implementation.cpp` | `svr_implementation` | Tiny linear problem, then an RBF fit to a noisy sine with its tube | `results/03_implementation_results/`: `predictions.csv`, `predictions.svg` |
| `04_end_to_end.cpp` | `svr_end_to_end` | Full project on diabetes: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid C in {1, 10, 100} (epsilon 0.1, RBF gamma 0.1) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `svr_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. The `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_svr (Model.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target svr_end_to_end svr_predict svr_tests
build\03_ml_course\01_supervised\01_regression\08_svr\svr_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\08_svr\svr_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\08_svr\svr_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `svr_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by svr_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R svr
```

`svr_numerical` runs `tests/model_test.cpp`. The fixture checks are listed in the table. Any failed check throws, so the exit code is non-zero. `svr_workflow` runs `svr_end_to_end --quick`. It fails if the pipeline or its reload verification fails.

## Key takeaways

- Points inside the tube contribute nothing to the solution: the model is sparse in the data.
- C is a regulariser in disguise: small C, wide tolerance, smoother fit.
- Kernel SVR scales as O(n^2) memory, which is why the course keeps n modest.

## Next module

`09_gaussian_process`: a kernel model that returns uncertainty as well as a mean.