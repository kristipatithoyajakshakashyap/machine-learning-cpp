# Data Science in C++17

## Purpose
This track builds a small data-science toolkit, `dsts`, and uses it in lesson
programs that mirror a pandas/numpy/scipy workflow without Python. The library
provides typed tabular data (`DataFrame`, `Series`, optional numeric cells),
CSV I/O, descriptive statistics, cleaning, joining, feature engineering,
sampling, SVG plots, time-series helpers, calendar utilities and PCA. Every
lesson `.cpp` compiles as its own executable that links `dsts` through CMake.

## Prerequisites
Track 01_get_to_know (C++17 basics through project structure and testing).

## Study order
Start with 00_math_foundations, then modules 01 through 10 and 13. Complete
12_advanced_eda before 11_final_pipeline. The final Titanic pipeline uses shared
transformations, a reproducible raw split, training-only learned preprocessing,
a transparent classifier, metrics and saved-state verification.

| Module | Targets | Topic |
|--------|---------|-------|
| `00_math_foundations` | `s00_math_01`..`s00_math_07` | vectors, projection, probability, sampling, gradients, optimisation, eigen/SVD |
| `01_series_dataframe` | `s01_*` | `Series`, `DataFrame`, CSV I/O |
| `02_cleaning` | `s02_*` | missing values, duplicates, outliers, scaling, one-hot |
| `03_eda` | `s03_*` | value counts, summaries, histograms, correlation |
| `04_plots` | `s04_*` | SVG line, scatter, bar, histogram, box, heat map |
| `05_joins_reshape` | `s05_*` | concat, merge, group-by, pivot |
| `06_feature_engineering` | `s06_*` | binning, quantiles, skew, log1p |
| `07_statistics_tests` | `s07_*` | correlations, t-tests, chi-square, bootstrap |
| `08_train_test` | `s08_*` | random and stratified splits, k-fold |
| `09_time_series` | `s09_*` | rolling mean, autocorrelation, differencing |
| `10_dimensionality` | `s10_*` | PCA fit and transform |
| `11_final_pipeline` | `s11_01_clean`..`s11_05_eval`, `s11_predict` | Titanic capstone |
| `12_advanced_eda` | `s12_eda_01`..`s12_eda_08` | disciplined EDA workflow and report |
| `13_datetime` | `s13_*` | date parsing, resampling, rolling windows |

Each module README lists its lessons, targets, outputs and the dsts functions used.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s11_05_eval
.\build\02_data_science\11_final_pipeline\s11_05_eval.exe
ctest --test-dir build -R "dsts_|s00_math|s12_eda" --output-on-failure
```

Executables land under `build/02_data_science/<module>/`. Every lesson reads
its dataset from `02_data_science/data/` (compile-time `DATA_DIR`) and writes
output files to a module-local `results/<source stem>_results/` directory
(compile-time `RUN_OUTPUT_DIR`, e.g. `04_csv_io.cpp` ->
`results/04_csv_io_results/`), created on first run.

## Library notes
`PcaResult.mean` stores fitted centering; `pca_transform` applies those training
axes to new rows. CSV serialization escapes quotes/newlines, retains numeric
precision and writes missing numeric values as empty cells. Schema inference
infers all-numeric text as numbers; use an explicit `DataFrame` schema where a
numeric-looking identifier must remain text.

Tests (`tests/regression.cpp`, ctest `dsts_regression`) exercise C++ numerical
fixtures, CSV round-trip, preprocessing isolation and model reload. See
`../docs/learning_path.md` for the study order and validation rules.

## Key takeaways
- A column store of optional doubles plus string columns is enough to teach
  the whole pandas workflow.
- Every learned quantity (median, mean, scale, PCA axis) is fitted on training
  rows and applied to the rest.
- Outputs are files you can diff; tests assert numbers, not screenshots.

## Next track
Continue with **03_ml_course**.
