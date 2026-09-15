# Categorical relationships

## Purpose
Builds the Sex x Pclass contingency table and tests independence with a chi-square test, reporting Cramer's V as an effect size so that significance is not confused with strength.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_04` | contingency tables, chi-square test of independence, Cramer's V | `sex_by_class.csv`, `categorical.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_04
.\build\02_data_science\12_advanced_eda\04_categorical_relationships\s12_eda_04.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_04`.

## dsts functions used
`read_csv`, `stratified_split`, `chi2_independence`, `DataFrame::strings`, `DataFrame::numeric`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- A small p-value says the association is unlikely to be chance; Cramer's V says how strong it is.
- Report both, and the cell counts they came from.

## Next module
Continue with **05_multivariate_analysis**.
