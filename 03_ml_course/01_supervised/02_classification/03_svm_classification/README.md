# Support Vector Machine Classification

## Problem and core idea

A support vector machine chooses the linear boundary with the largest margin to the closest training points and pays a hinge-loss penalty, scaled by C, for points inside or beyond the margin. Only the support vectors determine the boundary. With an RBF kernel the same idea fits curved boundaries. The course trains the linear SVM by sub-gradient descent on the primal and the kernel SVM through the dual.

## Dataset and why

`breast_cancer` for the linear lessons and the end-to-end project (C in {0.1, 1, 10}); it is nearly linearly separable, so the margin picture is faithful. Lesson 4 uses two petal features of `iris` so the RBF boundary can be understood in 2-D. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_logistic_regression` (compare hinge with log-loss), the notion of a margin, and dot products. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `SVM.hpp / SVM.cpp` | - | `ml::LinearSVM` and the RBF kernel SVM: hinge loss, `C`, decision function, save/load | - |
| `01_theory.cpp` | `csvm_theory` | Support vector machine in one page | prints only |
| `02_math_intuition.cpp` | `csvm_math_intuition` | Hinge loss and the RBF kernel, by hand | prints only |
| `03_implementation.cpp` | `csvm_implementation` | Linear SVM on breast cancer with a seeded split | `results/03_implementation_results/`: `c05_svm.split` |
| `04_rbf_kernel.cpp` | `csvm_04_rbf_kernel` | RBF SVM on two iris petal features | `results/04_rbf_kernel_results/`: `c06_svm.split` |
| `05_c_curve.cpp` | `csvm_05_c_curve` | Test accuracy versus C on breast cancer | `results/05_c_curve_results/`: `c_curve.svg` |
| `06_end_to_end.cpp` | `csvm_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid C in {0.1, 1, 10} (lr 0.01, 800 epochs) | `results/06_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `csvm_predict` | `#include`s `06_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_csvm (SVM.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target csvm_end_to_end csvm_predict
build\03_ml_course\01_supervised\02_classification\03_svm_classification\csvm_end_to_end            # full dataset -> results/06_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\03_svm_classification\csvm_end_to_end --quick    # seeded 400-row subset -> results/06_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\03_svm_classification\csvm_predict --predict results\06_end_to_end_results\full\data\holdout_features.csv --model results\06_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `csvm_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

## Results layout

Each executable owns `results/<source stem>_results/` inside this module (`03_implementation.cpp` owns `results/03_implementation_results/`, and so on). The project run writes:

```
results/06_end_to_end_results/
  full/   or   quick/
    data/          row_assignments.csv, holdout_features.csv, holdout.split
    eda/           summaries and figures computed on training rows only
    validation/    candidate_scores.csv, summary.csv, figures/parameter_curve.svg
    evaluation/    metrics.json, predictions.csv, per-class or residual files, figures/
    model/         model_state.txt, preprocessing_state.txt, target_state.txt, features.txt
    inference/     reload_verification.json
    run_manifest.json, report.md, execution.log
results/predict_results/predictions.csv        (written by csvm_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R csvm` matches nothing. The smoke check is `csvm_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Hinge loss is zero for confidently correct points, so most rows do not influence the final boundary.
- C is a regulariser: small C, wide margin, more bias; the accuracy curve is flat over a wide range.
- A linear SVM outputs a signed distance, not a probability; the pipeline's probability plots are approximate.

## Next module

`04_naive_bayes`: a generative model that needs no optimiser at all.
