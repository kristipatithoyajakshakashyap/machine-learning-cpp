# Module 08: Train/test splits and sampling

## Purpose
The split decisions that make a model's score believable. Toolkit added this
module: `include/dsts/sampling.hpp`. It provides `train_test_split()`,
`stratified_split()`, `sample_indices()`, `kfold_indices()` and
`stratified_kfold_indices()`, all seeded and reproducible.

## Prerequisites

Modules 02_cleaning and 07_statistics_tests (bootstrap, variability).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_random_split.cpp` | `s08_01_split` | Deterministic hold-out at several sizes. Data: titanic.csv (real data, 891 passengers) | `train_test_split(random_state=42)` | prints only |
| `02_stratified.cpp` | `s08_02_strat` | Keep the outcome mix in both splits. Data: titanic.csv (891 passengers, survival rate 342/891 = 0.384) | `StratifiedShuffleSplit` | prints only |
| `03_sampling_bias.cpp` | `s08_03_bias` | "First 40 rows" vs a random 40. Data: tips.csv (real data, 244 restaurant tips) | `.sample(seed=1)` | prints only |
| `04_boot_sample_ci.cpp` | `s08_04_boot` | Seeded indices and a bootstrap CI of a correlation. Data: tips.csv (real data, 244 restaurant tips) | `rng.choice(..., replace=True)` | prints only |
| `05_eval_split_size.cpp` | `s08_05_size` | Split size and seed stability (the k-fold idea). Data: tips.csv (real data, 244 restaurant tips) | cross-validation | prints only |
| `06_cross_validation.cpp` | `s08_06_cv` | 5-fold and stratified 5-fold on iris, nearest-class-mean accuracy per fold, mean +/- std (`folds.csv`, `fold_scores.csv`). Data: iris.csv (150 rows, 3 species x 50). Model: nearest class mean on the four measurements (fit = per-class means, predict = closest mean)  | `KFold`, `StratifiedKFold`, `cross_val_score` | `fold_scores.csv`, `folds.csv` under `results/06_cross_validation_results/` |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/08_train_test/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s08_01_split
.\build\02_data_science\08_train_test\s08_01_split.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Series`.

- `01_random_split.cpp`: `fmt`, `read_csv`, `train_test_split`
- `02_stratified.cpp`: `fmt`, `read_csv`, `stratified_split`, `train_test_split`
- `03_sampling_bias.cpp`: `read_csv`, `train_test_split`
- `04_boot_sample_ci.cpp`: `fmt`, `pearson`, `read_csv`, `sample_indices`
- `05_eval_split_size.cpp`: `fmt`, `read_csv`, `train_test_split`
- `06_cross_validation.cpp`: `fmt`, `kfold_indices`, `read_csv`, `stratified_kfold_indices`, `write_csv`

## Key ideas
- One seed, one split, forever. Same arguments reproduce the same rows
  (lesson 1 prints identical splits for `seed=7`).
- Stratification kills drift. Unstratified test sets drifted to 0.40
  survival while the stratified test stayed at 0.384, matching the full data.
- Ordering is a bias. The first 40 tips rows contain zero Thursdays. Any
  random 40 contain ~10.
- Re-splitting shows the wobble. Across five seeds the test mean tip spans
  2.89 to 3.05 at 60/40 and 2.78 to 3.09 at 80/20. That spread is what k-fold averages out.
- k-fold tests every row once. Plain folds score 0.92 +/- 0.038 on iris,
  stratified folds hold exactly 10 of each species, so no fold misses a class.

## Key takeaways
- A seeded split is reproducible. The same arguments always give the same rows.
- Stratify so the outcome mix survives the split. Never take the first N rows.
- k-fold cross-validation averages out the wobble of a single split.

## Next module

09_time_series - trend, seasonality, baseline forecasts and autocorrelation.