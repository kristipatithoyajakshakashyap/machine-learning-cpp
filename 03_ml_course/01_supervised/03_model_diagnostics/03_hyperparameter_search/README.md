# Hyper-parameter Search

## Problem and core idea

Most models have several knobs, and the best combination is found by searching a space with cross-validation. Grid search is exhaustive. Random search samples the space and often finds a comparable point far cheaper. Either way the best CV score is optimistically biased because it was selected, and nested cross-validation measures that bias by re-running the search inside each outer fold. The end-to-end model maps the pipeline's one scalar onto a 2-D forest grid so the standard pipeline performs a full grid search.

## Dataset and why

`wine` (178 x 13, 3 classes) with a 100-tree forest. The search space is max_depth {4, 8, 16} x min_leaf {1, 2, 5}. On wine the differences are small, which is precisely when selection bias is easy to mistake for real improvement. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`01_feature_importance_and_learning_curves` (the forest), cross-validation, and basic probability for the bias estimate. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp` | - | `course::TunedForest(p)`: index 0..8 decodes to (max_depth, min_leaf). It forwards to the forest. Save/load tag `TunedForest_V1` | - |
| `helper/eval/search.hpp` | - | `ParamSpace`, `grid_search`, `random_search`, `nested_cv` | - |
| `tests/model_test.cpp` | `hps_tests` | fixture test: grid search visits all 10 cells and matches a brute-force optimum, random search draws distinct cells and is capped at the space size, nested CV yields one row per outer fold with a finite naive score, index 7 decodes to depth 8 / leaf 5, save/load is bit-exact, index 9 is rejected | prints only |
| `01_theory.cpp` | `hps_theory` | Grid, random and nested search explained | prints only |
| `02_math_intuition.cpp` | `hps_math_intuition` | The optimistic bias of a selected maximum, in closed form | prints only |
| `03_implementation.cpp` | `hps_implementation` | Grid versus random versus nested CV on wine | `results/03_implementation_results/`: `grid_search.csv`, `random_search.csv`, `nested_cv.csv`, `search_comparison.svg` |
| `04_end_to_end.cpp` | `hps_end_to_end` | Full project on wine: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid grid index in {0..8} = max_depth {4, 8, 16} x min_leaf {1, 2, 5}. Then it appends `validation/random_search.csv`, `validation/nested_cv.csv`, `validation/figures/search_comparison.svg` | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `hps_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. The `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_tree_models (Model.hpp is header-only)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target hps_end_to_end hps_predict hps_tests
build\03_ml_course\01_supervised\03_model_diagnostics\03_hyperparameter_search\hps_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\03_model_diagnostics\03_hyperparameter_search\hps_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\03_model_diagnostics\03_hyperparameter_search\hps_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `hps_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by hps_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R hps
```

`hps_numerical` runs `tests/model_test.cpp`. The fixture checks are listed in the table. Any failed check throws, so the exit code is non-zero. `hps_workflow` runs `hps_end_to_end --quick`. It fails if the pipeline or its reload verification fails.

## Key takeaways

- The best CV score is not an estimate of generalisation. The nested CV mean is.
- Random search with a few draws usually lands within noise of the grid optimum.
- Encode multi-dimensional grids behind one index when a pipeline only tunes one scalar.

## Next module

`../04_ensembles/01_voting_and_stacking`: combining tuned models.