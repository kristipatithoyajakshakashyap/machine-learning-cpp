# Quadratic Discriminant Analysis

## Problem and core idea

QDA is LDA with one covariance per class, so the decision boundaries become quadratic surfaces. It can model classes with different spreads and orientations, at the price of p(p+1)/2 covariance parameters per class, which makes regularisation essential on small or wide data. The course derives `course::QDA` from the LDA class by switching on class-specific covariance, so every numerical trick (Cholesky factors, log-determinants, log-sum-exp) is shared.

## Dataset and why

A seeded 2-D three-class sample whose classes have different diagonal covariances in the lesson: the case LDA cannot fit. The end-to-end project uses `breast_cancer` with ridge in {0.001, 0.01, 0.1}, where 30 features and two classes make the per-class covariance estimate noisy enough that the ridge choice matters. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`11_lda` (this module reuses its implementation) and the multivariate Gaussian density. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp / Model.cpp` | - | `course::QDA`: `LDA` with `quadratic=true`, save/load tag `QDA_V1` | - |
| `tests/model_test.cpp` | `qda_tests` | fixture test: same checks as LDA (perfect separable fit, posteriors sum to one, round trip, empty input rejected) through the QDA tag | prints only |
| `01_theory.cpp` | `qda_theory` | Why per-class covariance gives quadratic boundaries | prints only |
| `02_math_intuition.cpp` | `qda_math_intuition` | Two classes with different variances, by hand | prints only |
| `03_implementation.cpp` | `qda_implementation` | 1-D check, seeded 2-D sample with different spreads, recovered class covariances | `results/03_implementation_results/`: `discriminant_scores.csv`, `class_covariance.csv`, `scatter.svg` |
| `04_end_to_end.cpp` | `qda_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid ridge in {0.001, 0.01, 0.1} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `qda_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; the `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_qda (Model.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target qda_end_to_end qda_predict qda_tests
build\03_ml_course\01_supervised\02_classification\12_qda\qda_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\12_qda\qda_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\12_qda\qda_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `qda_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by qda_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R qda
```

`qda_numerical` runs `tests/model_test.cpp` (the fixture checks listed in the table; any failed check throws, so the exit code is non-zero). `qda_workflow` runs `qda_end_to_end --quick` and fails if the pipeline or its reload verification fails.

## Key takeaways

- QDA wins when class spreads differ; LDA wins when data are scarce. Compare `validation/summary.csv` of both modules.
- The number of covariance parameters grows quadratically with features: regularise.
- Inheritance made QDA a few lines; the numerical care lives in one place.

## Next module

`13_imbalanced_classification`: when one class is rare.
