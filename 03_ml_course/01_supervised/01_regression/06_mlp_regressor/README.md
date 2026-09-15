# MLP Regressor

## Problem and core idea

A multi-layer perceptron stacks affine maps and non-linear activations, and learns every weight by gradient descent on the training loss using backpropagation. With one hidden layer it can approximate any continuous function given enough units, but it needs standardised inputs, a sensible learning rate and enough epochs. It is the only model in this group whose training is itself an iterative optimisation with a seed.

## Dataset and why

California housing in both the lesson and the end-to-end project; the 8 continuous features standardise well and there are enough rows for a 16-8 network to learn something a linear model cannot. The learning-rate grid {0.001, 0.005, 0.01} shows how sensitive the optimiser is. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`; the build passes their folder as `DATA_DIR`.

## Prerequisites

`01_linear_regression` (the MLP with no hidden layer is OLS), the chain rule, and gradient descent from `02_math_intuition.cpp` of that module. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `MLP.hpp / MLP.cpp` | - | `ml::MLP`: dense layers with ReLU, mini-batch SGD, regression and softmax heads, save/load (reused by `02_classification/09_mlp_classifier`) | - |
| `01_theory.cpp` | `rmlp_theory` | The multi-layer perceptron in one page | prints only |
| `02_math_intuition.cpp` | `rmlp_math_intuition` | One backpropagation step, by hand | prints only |
| `03_implementation.cpp` | `rmlp_implementation` | MLP versus ridge on a California subsample | `results/03_implementation_results/`: `s10_mlp.split` |
| `04_end_to_end.cpp` | `rmlp_end_to_end` | Full project on California housing: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check; parameter grid learning rate in {0.001, 0.005, 0.01} (layers 16-8, 150 epochs) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `rmlp_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`; wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above; module library `ml_rmlp (MLP.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target rmlp_end_to_end rmlp_predict
build\03_ml_course\01_supervised\01_regression\06_mlp_regressor\rmlp_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\01_regression\06_mlp_regressor\rmlp_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\01_regression\06_mlp_regressor\rmlp_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds; run without it for the real numbers. `rmlp_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by rmlp_predict)
```

`results/` folders are generated and can be deleted at any time; nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R rmlp` matches nothing. The smoke check is `rmlp_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails; the repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Standardisation is not optional for gradient-trained models.
- Training loss going down does not mean the holdout score goes up: keep the validation curve in view.
- The seed is part of the model; two runs with different seeds are two different models.

## Next module

`07_elastic_net`: back to linear models with a penalty that mixes L1 and L2.
