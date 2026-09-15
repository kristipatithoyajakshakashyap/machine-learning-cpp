# EDA report

## Purpose
Runs every previous stage in one process and writes a combined report that links the tables and figures to concrete modelling recommendations (training-fitted imputation, standardisation, explicit encoding, majority baseline).

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_08` | assembling all stages into a reproducible report | all stage-1..7 files plus `report.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_08
.\build\02_data_science\12_advanced_eda\08_eda_report\s12_eda_08.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_08`.

## dsts functions used
everything used by stages 1-7: `read_csv`, `stratified_split`, `duplicated`, `outlier_mask`, `chi2_independence`, `correlation_matrix`, `pearson`, `spearman`, `group_mean`, `write_svg_histogram`, `write_svg_heatmap`, `write_svg_line`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- An EDA report should end in actions for the modelling pipeline.
- Every number in the report comes from the training partition only.

## Next module
Continue with **11_final_pipeline** (the Titanic capstone).
