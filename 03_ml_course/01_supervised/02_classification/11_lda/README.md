# Linear Discriminant Analysis

## Problem and core idea

LDA models each class as a Gaussian with its own mean but one pooled covariance, and classifies by the largest posterior. Because the covariance is shared, the quadratic terms cancel and the decision boundaries are linear. The implementation stores a Cholesky factor of the regularised covariance instead of an inverse: a triangular solve gives the Mahalanobis distance and the log-determinant falls out of the diagonal. A ridge term keeps near-singular covariances factorisable.

## Dataset and why

Generated 1-D and 2-D Gaussian fixtures in the lessons (the 2-D sample obeys the shared-covariance assumption exactly). The end-to-end project uses `breast_cancer` with ridge in {1e-4, 0.01, 0.1}; 30 correlated features make the regularisation grid meaningful. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

Multivariate Gaussians, covariance matrices, Bayes' rule (`04_naive_bayes`), and the Cholesky factorisation. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp / Model.cpp` | - | `course::LDA`: means, priors, pooled covariance as a Cholesky factor, log-sum-exp posteriors, save/load; `quadratic=true` turns it into QDA | - |
| `tests/model_test.cpp` | `lda_tests` | fixture test: separable 1-D data classified perfectly with non-consecutive labels, posteriors sum to one, save/load round trip, empty input rejected | prints only |
| `01_theory.cpp` | `lda_theory` | Gaussian classes with a shared covariance | prints only |
| `02_math_intuition.cpp` | `lda_math_intuition` | Discriminant scores on a tiny fixture, by hand | prints only |
| `03_implementation.cpp` | `lda_implementation` | 1-D sanity check, then a seeded 2-D three-class sample | `results/03_implementation_results/`: `discriminant_scores.csv`, `scatter.svg` |
| `04_end_to_end.cpp` | `lda_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid ridge in {1e-4, 0.01, 0.1} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `lda_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; the `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_lda (Model.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target lda_end_to_end lda_predict lda_tests
build\03_ml_course\01_supervised\02_classification\11_lda\lda_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\11_lda\lda_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\11_lda\lda_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `lda_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by lda_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R lda
```

`lda_numerical` runs `tests/model_test.cpp` (the fixture checks listed in the table; any failed check throws, so the exit code is non-zero). `lda_workflow` runs `lda_end_to_end --quick` and fails if the pipeline or its reload verification fails.

## Key takeaways

- Shared covariance gives linear boundaries and needs far fewer parameters than QDA.
- Never invert a covariance: factorise once and solve.
- Log-sum-exp turns discriminant scores into probabilities without overflow.

## Next module

`12_qda`: drop the shared-covariance assumption.
