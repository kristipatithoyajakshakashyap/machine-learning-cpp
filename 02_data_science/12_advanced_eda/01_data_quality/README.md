# Data quality

## Purpose
First pass over the training rows: column types, missing counts, duplicate rows and range sanity checks on Age and Fare. The stage throws if a numeric column contains negative or non-finite values, so a bad dataset fails loudly.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_01` | per-column type and missing fraction, duplicate detection, range checks | `quality.csv`, `quality.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_01
.\build\02_data_science\12_advanced_eda\01_data_quality\s12_eda_01.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_01`.

## dsts functions used
`read_csv`, `stratified_split`, `duplicated`, `DataFrame::columns`, `DataFrame::series`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- Profile types and missingness before any modelling decision.
- Fail fast on impossible values instead of silently imputing them.

## Next module
Continue with **02_missingness_analysis**.
