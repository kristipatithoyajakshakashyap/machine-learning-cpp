# Module 12: Advanced exploratory analysis

## Purpose
Eight runnable lessons that turn exploratory analysis into a disciplined,
reproducible workflow: data quality, missingness, outliers, categorical and
multivariate association, target relationships and temporal structure, with a
final report combining their tables and figures. Titanic analysis uses training
rows only. AirPassengers demonstrates chronological structure. No geographic
dataset is fabricated. That lesson explains applicability and spatial validation.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test.

## Lessons

Every lesson's `01_analysis.cpp` is a one-line driver calling
`advanced_eda::run(stage)`. The shared implementation lives in
`eda_workflow.hpp`. Outputs go to `results/01_analysis_results/` inside each
lesson folder.

| Lesson | Target | What it teaches | What it writes |
|--------|--------|-----------------|----------------|
| `01_data_quality` | `s12_eda_01` | types, missing counts, duplicates, range checks | `quality.csv`, `quality.md` |
| `02_missingness_analysis` | `s12_eda_02` | missing rate by group, MCAR/MAR/MNAR | `age_missing_by_sex.csv`, `missingness.md` |
| `03_outliers_and_distributions` | `s12_eda_03` | quantiles, IQR fences, histograms | `quantiles_outliers.csv`, `Age.svg`, `Fare.svg`, `outliers.md` |
| `04_categorical_relationships` | `s12_eda_04` | contingency table, chi-square, Cramer's V | `sex_by_class.csv`, `categorical.md` |
| `05_multivariate_analysis` | `s12_eda_05` | correlation heat map, Pearson vs Spearman | `correlation.svg`, `pairwise_association.csv` |
| `06_target_relationships` | `s12_eda_06` | survival rate by Sex with Wilson intervals | `survival_by_sex.csv`, `target.md` |
| `07_temporal_and_geographic_analysis` | `s12_eda_07` | lag-12 diagnostics on AirPassengers | `temporal_diagnostics.csv`, `passengers.svg`, `temporal.md` |
| `08_eda_report` | `s12_eda_08` | all stages plus a combined report | all of the above plus `report.md` |

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_08
.\build\02_data_science\12_advanced_eda\08_eda_report\s12_eda_08.exe
```

Each target reads `titanic.csv` (and `air_passengers.csv` for stage 7) from
`02_data_science/data/` via `DATA_DIR` and is registered as a ctest
(`ctest --test-dir build -R s12_eda`).

## dsts functions used
`read_csv`, `stratified_split`, `duplicated`, `outlier_mask`,
`chi2_independence`, `correlation_matrix`, `pearson`, `spearman`,
`group_mean`, `write_svg_histogram`, `write_svg_heatmap`, `write_svg_line`,
plus `DataFrame::series/numeric/strings/select_columns` and `Series::quantile`.

## Key takeaways
- Split first, explore the training partition, and keep the hold-out untouched.
- Pair every statistic with its sample size and an effect size.
- End exploration with concrete, testable preprocessing decisions.

## Next module
Continue with 11_final_pipeline, which applies these decisions end to end.