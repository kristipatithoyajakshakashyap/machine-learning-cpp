# Missingness analysis

## Purpose
Asks whether Age is missing at random by comparing the missing rate between Sex groups. Unequal rates are evidence against missing-completely-at-random; the lesson explains why the reverse cannot be proven from observed data alone.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_02` | missing-rate comparison across groups, MCAR/MAR/MNAR vocabulary | `age_missing_by_sex.csv`, `missingness.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_02
.\build\02_data_science\12_advanced_eda\02_missingness_analysis\s12_eda_02.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_02`.

## dsts functions used
`read_csv`, `stratified_split`, `DataFrame::strings`, `DataFrame::series`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- Compare missing rates across observed groups before choosing an imputer.
- Fit imputers on the training partition only.

## Next module
Continue with **03_outliers_and_distributions**.
