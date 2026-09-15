# Elastic Net

## Problem and core idea

Elastic net minimises squared error plus a mixed penalty: alpha times (l1_ratio * L1 + (1 - l1_ratio)/2 * L2). The L1 part produces exact zeros (feature selection), the L2 part keeps correlated features together instead of picking one arbitrarily. It is solved by cyclic coordinate descent with a soft-threshold update, which is fast because each coordinate update is closed form.

## Dataset and why

Small exact fixtures in the lessons so the coefficients can be checked against known values. The end-to-end project uses `diabetes` (442 x 10): its features are correlated, which is exactly where elastic net behaves better than plain lasso. Grid: alpha in {0.001, 0.01, 0.1, 1} at l1_ratio 0.5. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_linear_regression` (ridge and lasso), the soft-threshold operator, and coordinate descent. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp / Model.cpp` | - | `course::ElasticNet`: coordinate descent, `coef`/`intercept`, save/load | - |
| `tests/model_test.cpp` | `elastic_tests` | fixture test: exact recovery of a known line, zeroing under a strong penalty, save/load round trip, rejection of empty input | prints only |
| `01_theory.cpp` | `elastic_theory` | The penalty and why it mixes L1 and L2 | prints only |
| `02_math_intuition.cpp` | `elastic_math_intuition` | One soft-threshold coordinate update, by hand | prints only |
| `03_implementation.cpp` | `elastic_implementation` | Exact tiny problem plus a synthetic sparse problem; coefficient plot | `results/03_implementation_results/`: `coefficients.csv`, `coefficients.svg` |
| `04_end_to_end.cpp` | `elastic_end_to_end` | Full project on diabetes: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid alpha in {0.001, 0.01, 0.1, 1} (l1_ratio 0.5) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `elastic_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; the `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_elastic (Model.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target elastic_end_to_end elastic_predict elastic_tests
build\03_ml_course\01_supervised\01_regression\07_elastic_net\elastic_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\07_elastic_net\elastic_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\07_elastic_net\elastic_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `elastic_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by elastic_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R elastic
```

`elastic_numerical` runs `tests/model_test.cpp` (the fixture checks listed in the table; any failed check throws, so the exit code is non-zero). `elastic_workflow` runs `elastic_end_to_end --quick` and fails if the pipeline or its reload verification fails.

## Key takeaways

- Coordinate descent on a penalised least-squares objective converges in a handful of sweeps.
- The L2 part 'groups' correlated features; the L1 part sets the rest to zero.
- Compare `validation/summary.csv` with lasso and ridge from module 01 on the same data.

## Next module

`08_svr`: a loss that ignores small errors entirely.
