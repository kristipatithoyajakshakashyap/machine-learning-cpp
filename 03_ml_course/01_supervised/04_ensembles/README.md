# Ensembles

Combine models introduced earlier into one predictor. Every module in this group follows the same shape: a `theory.md` / `math_intuition.md` / `implementation.md` / `exercises.md` set, numbered lesson executables, an end-to-end project built on the shared pipeline in `helper/pipeline/supervised.hpp`, and a `predict` target that serves a saved run. The datasets are `wine` (178 x 13, 3 classes).

| Module | Target prefix | What it teaches |
|---|---|---|
| [`01_voting_and_stacking`](01_voting_and_stacking/README.md) | `ens_*` | Hard voting, soft voting and stacking with out-of-fold meta-features |

## Build and run

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course
cmake --build --preset course --target <prefix>_end_to_end <prefix>_predict
build\03_ml_course\01_supervised\04_ensembles\<module>\<prefix>_end_to_end            # full dataset
build\03_ml_course\01_supervised\04_ensembles\<module>\<prefix>_end_to_end --quick    # seeded 400-row subset
build\03_ml_course\01_supervised\04_ensembles\<module>\<prefix>_predict --predict <holdout_features.csv> --model <run>/model
```

Every executable writes into `results/<source stem>_results/` inside its own module; the project run adds `full/` or `quick/` below that. Modules with a `tests/` folder register `<prefix>_numerical` and `<prefix>_workflow` in CTest (`ctest --preset course -R <prefix>`). Each module README lists its files, outputs, tests and takeaways.

## Next group

`../../02_unsupervised`
