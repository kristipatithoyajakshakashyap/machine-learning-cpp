# Voting and Stacking

## Problem and core idea

Different models make different mistakes, so combining them often beats the best single one. Hard voting counts class labels, soft voting averages probabilities, and stacking trains a meta-learner on the members' probabilities. The catch: the meta-learner must see out-of-fold predictions, not in-sample ones, or it learns to trust whichever member overfits the most. One class exposes all three modes behind one parameter so the standard pipeline compares them on equal footing.

## Dataset and why

`wine` (178 rows, 13 features, 3 classes) with multinomial logistic regression, 5-NN and a 50-tree forest as members. Every member is strong on wine, so the three modes differ within one standard deviation. The point is the procedure, not a huge win. Datasets are CSV files under `03_ml_course/helper/data/` and are loaded through `helper/data/datasets.hpp`. The build passes their folder as `DATA_DIR`.

## Prerequisites

`02_classification/01_logistic_regression`, `02_classification/02_knn_classification` and `01_regression/04_random_forest_regressor` (the members, with the forest in classification mode), plus k-fold cross-validation from `02_classification/10_model_evaluation`. Read `theory.md`, `math_intuition.md` and `implementation.md`, run the numbered lessons in order, then work through `exercises.md`.

## Files

| File | Target | Purpose | Outputs |
|---|---|---|---|
| `theory.md` | - | Assumptions and the conceptual picture | - |
| `math_intuition.md` | - | Worked formulas and numeric examples | - |
| `implementation.md` | - | How the C++ code carries out the workflow | - |
| `exercises.md` | - | Practice tasks and acceptance evidence | - |
| `Model.hpp` | - | `course::VotingStacking(mode)`: mode 0 hard vote, 1 soft vote, 2 stacking with a multinomial meta-learner on 5-fold out-of-fold probabilities (seed 42). Save/load tag `VotingStacking_V1` | - |
| `tests/model_test.cpp` | `ens_tests` | fixture test: hard vote ties break in a fixed order, soft vote ranks all members in a known ordering, out-of-fold probabilities are held out per fold, the stacked version wins on a benchmark sample, round trip is exact, an unknown mode throws | prints only |
| `01_theory.cpp` | `ens_theory` | Voting, averaging and stacking concepts | prints only |
| `02_math_intuition.cpp` | `ens_math_intuition` | One soft-vote and one stacking decision, by hand | prints only |
| `03_implementation.cpp` | `ens_implementation` | Hard, soft and stacked versions on the three members, plus the stacking meta-learner weights | `results/03_implementation_results/`: `comparison.csv`, `macro_f1.svg`, `meta_weights.csv` |
| `04_end_to_end.cpp` | `ens_end_to_end` | Full project on wine: `ml::run_supervised` (helper/pipeline/supervised.hpp): stratified/seeded 80/20 holdout, EDA on training rows, training-only k-fold cross-validation over the parameter grid, final fit, holdout evaluation with bootstrap intervals, model persistence and reload check. Parameter grid ensemble mode in {0, 1, 2} | `results/04_end_to_end_results/full/` or `quick/` (see below) |
| `predict.cpp` | `ens_predict` | `#include`s `04_end_to_end.cpp` so the same code serves inference, compiled with its own `RUN_OUTPUT_DIR`. The `--predict`/`--model` pair switches that `main` to inference mode, so always pass both | `results/predict_results/predictions.csv` |
| `CMakeLists.txt` | - | Registers the targets above. Module library `ml_ens (Model.hpp is header-only)` | - |

## Build and run

From the repository root with the MinGW toolchain on PATH:

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target ens_end_to_end ens_predict ens_tests
build\03_ml_course\01_supervised\04_ensembles\01_voting_and_stacking\ens_end_to_end            # full dataset -> results/04_end_to_end_results/full/
build\03_ml_course\01_supervised\04_ensembles\01_voting_and_stacking\ens_end_to_end --quick    # seeded 400-row subset -> results/04_end_to_end_results/quick/
build\03_ml_course\01_supervised\04_ensembles\01_voting_and_stacking\ens_predict --predict results\04_end_to_end_results\full\data\holdout_features.csv --model results\04_end_to_end_results\full\model
```

`--quick` keeps a seeded random subset of at most 400 rows so the whole workflow finishes in seconds. Run without it for the real numbers. `ens_predict` reads any CSV that contains the feature columns named in `model/features.txt` (the pipeline's own `data/holdout_features.csv` is the ready-made example), restores the model, preprocessing and target transform from `<run>/model`, and writes `row_id,prediction` rows. Every other lesson target runs without arguments.

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
results/predict_results/predictions.csv        (written by ens_predict)
```

`results/` folders are generated. Delete them at any time. Nothing in the build depends on them.

## Tests

```powershell
ctest --preset course -R ens
```

`ens_numerical` runs `tests/model_test.cpp`. The fixture checks are listed in the table. Any failed check throws, so the exit code is non-zero. `ens_workflow` runs `ens_end_to_end --quick`. It fails if the pipeline or its reload verification fails.

## Key takeaways

- Stacking earns its keep only with out-of-fold probabilities, never in-sample ones.
- Members that are strongly correlated add little together.
- Everything stays a pipeline model: one scalar parameter chooses the mode.

## Next group

`../../02_unsupervised`