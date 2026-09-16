# Feature Selection

## Problem and core idea

Fewer inputs mean cheaper, more robust and more interpretable models, if the dropped inputs carried nothing the kept ones lack. This module compares three families: filter methods that score each column alone (ANOVA F), wrapper methods that add columns greedily by cross-validated score (forward selection), and embedded methods that let a penalised model choose (the L1 path). The end-to-end model wraps filter selection plus a forest so the number of kept columns is tuned by the pipeline.

## Dataset and why

`breast_cancer` (30 correlated features, 2 classes). Many columns are near-duplicates (mean, error and worst of the same measurement), so a handful of them recover almost all of the accuracy, which makes the selection curves easy to read. Grid: k in {5, 10, 20, 30}. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_feature_importance_and_learning_curves`, lasso from `01_regression/01`, and stratified cross-validation. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp` | - | `course::SelectedForest(k)`: ANOVA filter keeps the top-k columns, then a forest on those. Save/load stores the kept indices | - |
| `helper/eval/feature_selection.hpp` | - | `filter_scores`, `top_k`, `forward_selection`, `l1_path`, `l1_scores` | - |
| `tests/model_test.cpp` | `fsel_tests` | fixture test: ANOVA F orders signal > mild > noise, forward selection picks the signal column first with > 95% CV accuracy, the L1 path keeps the signal column longest, `SelectedForest(2)` fits and round-trips exactly through save/load, k > columns and empty data throw | prints only |
| `01_theory.cpp` | `fsel_theory` | Filter, wrapper and embedded selection | prints only |
| `02_math_intuition.cpp` | `fsel_math_intuition` | ANOVA F and one forward-selection step, by hand | prints only |
| `03_implementation.cpp` | `fsel_implementation` | All three families on breast cancer with rankings | `results/03_implementation_results/`: `rankings.csv`, `rankings.svg`, `l1_path.csv` |
| `04_end_to_end.cpp` | `fsel_end_to_end` | Full project on breast cancer: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid kept features k in {5, 10, 20, 30}. Then it appends `validation/filter_scores.csv`, `validation/forward_selection.csv`, `validation/figures/forward_selection.svg` (training rows only) | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `fsel_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. The `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_fsel (Model.hpp is header-only)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target fsel_end_to_end fsel_predict fsel_tests
build\03_ml_course\01_supervised\03_model_diagnostics\02_feature_selection\fsel_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\03_model_diagnostics\02_feature_selection\fsel_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\03_model_diagnostics\02_feature_selection\fsel_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `fsel_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by fsel_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R fsel
```

`fsel_numerical` runs `tests/model_test.cpp`. The fixture checks are listed in the table. Any failed check throws, so the exit code is non-zero. `fsel_workflow` runs `fsel_end_to_end --quick`. It fails if the pipeline or its reload verification fails.

## Key takeaways

- Filters are cheap but blind to interactions. Wrappers see interactions but cost a CV per step.
- Select on training rows only, inside the CV, or the holdout score is contaminated.
- k is a hyper-parameter: let cross-validation pick it rather than a fixed threshold.

## Next module

`03_hyperparameter_search`: tuning more than one knob honestly.