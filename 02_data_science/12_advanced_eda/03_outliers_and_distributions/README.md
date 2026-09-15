# Outliers and distributions

## Purpose
Computes the five-number summary and IQR outlier count for Age and Fare and draws one histogram per feature. Fare is strongly right-skewed, which motivates comparing median to mean before choosing a transformation.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_03` | quantiles, IQR fences, histograms, skew | `quantiles_outliers.csv`, `Age.svg`, `Fare.svg`, `outliers.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_03
.\build\02_data_science\12_advanced_eda\03_outliers_and_distributions\s12_eda_03.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_03`.

## dsts functions used
`read_csv`, `stratified_split`, `outlier_mask`, `Series::quantile`, `write_svg_histogram`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- IQR fences flag unusual values, not errors; investigate before deleting.
- Skewed features often need a log or rank transform.

## Next module
Continue with **04_categorical_relationships**.
