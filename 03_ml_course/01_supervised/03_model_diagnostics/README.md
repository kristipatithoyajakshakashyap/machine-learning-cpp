# Model Diagnostics

Interrogate a fitted model instead of adding a new learner. Every module in this group follows the same shape: a `theory.md` / `math_intuition.md` / `implementation.md` / `exercises.md` set, numbered lesson executables, an end-to-end project built on the shared pipeline in `helper/pipeline/supervised.hpp`, and a `predict` target that serves a saved run. The datasets are `wine` for the forest-based modules, `breast_cancer` for feature selection.

| Module | Target prefix | What it teaches |
|---|---|---|
| [`01_feature_importance_and_learning_curves`](01_feature_importance_and_learning_curves/README.md) | `diag_*` | Permutation importance and learning curves |
| [`02_feature_selection`](02_feature_selection/README.md) | `fsel_*` | Filter, wrapper and embedded selection; tuned number of kept columns |
| [`03_hyperparameter_search`](03_hyperparameter_search/README.md) | `hps_*` | Grid, random and nested cross-validation |

## Build and run

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target <prefix>_end_to_end <prefix>_predict
build\03_ml_course\01_supervised\03_model_diagnostics\<module>\<prefix>_end_to_end            # full dataset
build\03_ml_course\01_supervised\03_model_diagnostics\<module>\<prefix>_end_to_end --quick    # seeded 400-row subset
build\03_ml_course\01_supervised\03_model_diagnostics\<module>\<prefix>_predict --predict <holdout_features.csv> --model <run>/model
```

Every executable writes into `results/<source stem>_results/` inside its own module; the project run adds `full/` or `quick/` below that. Modules with a `tests/` folder register `<prefix>_numerical` and `<prefix>_workflow` in CTest (`ctest --preset course -R <prefix>`). Each module README lists its files, outputs, tests and takeaways.

## Next group

`../04_ensembles/01_voting_and_stacking`
