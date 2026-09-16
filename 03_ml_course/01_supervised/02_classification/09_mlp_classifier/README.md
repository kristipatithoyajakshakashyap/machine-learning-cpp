# MLP Classifier

## Problem and core idea

A multi-layer perceptron with a softmax output layer learns non-linear features in its hidden layers and a linear classifier on top, all by backpropagation of the cross-entropy loss. It is the most flexible model in the group and the most sensitive to scaling, learning rate and epochs. The classification MLP forwards to the regression module's network with a softmax head.

## Dataset and why

`wine` (3 classes) in the lesson so the softmax head is exercised. The end-to-end project uses `breast_cancer` with learning rate in {0.001, 0.005, 0.01} for a 16-8 network trained 200 epochs. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_regression/06_mlp_regressor`, softmax and cross-entropy from `01_logistic_regression`. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `MLP.hpp / MLP.cpp` | - | `ml::MLP` with a softmax head: forwards to the regression module's layers, `predict_proba`, save/load | - |
| `01_theory.cpp` | `cmlp_theory` | Neural network classification in one page | prints only |
| `02_math_intuition.cpp` | `cmlp_math_intuition` | Softmax and one backpropagation step, by hand | prints only |
| `03_implementation.cpp` | `cmlp_implementation` | MLP on wine with a seeded split | `results/03_implementation_results/`: `c10_mlp.split` |
| `04_end_to_end.cpp` | `cmlp_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid learning rate in {0.001, 0.005, 0.01} (layers 16-8, 200 epochs) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `cmlp_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. It wraps it in a `main` that refuses to run without `--predict` | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_cmlp (MLP.cpp)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target cmlp_end_to_end cmlp_predict
build\03_ml_course\01_supervised\02_classification\09_mlp_classifier\cmlp_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\02_classification\09_mlp_classifier\cmlp_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\02_classification\09_mlp_classifier\cmlp_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `cmlp_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by cmlp_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

This module registers no CTest entries, so `ctest --preset course -R cmlp` matches nothing. The smoke check is `cmlp_end_to_end --quick`, which exits non-zero if any stage or the reload verification fails. The repository-wide numerical tests under `03_ml_course/tests/` cover the shared helpers it uses.

## Key takeaways

- Softmax plus cross-entropy has the same gradient form as the sigmoid plus log-loss: (p - y).
- On small tabular data an MLP rarely beats a forest. It wins when features must be learned.
- Standardise, seed, and watch the validation curve: three rules for any gradient-trained model.

## Next module

`10_model_evaluation`: how to compare all of the above honestly.