# Multivariate analysis

## Purpose
Looks at all numeric predictors at once: a Pearson correlation heat map plus a pairwise table with complete-pair counts, Pearson and Spearman coefficients, so rank-based and linear association can be compared.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_05` | correlation matrix, heat map, Pearson vs Spearman, complete-pair counts | `correlation.svg`, `pairwise_association.csv` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver; the worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_05
.\build\02_data_science\12_advanced_eda\05_multivariate_analysis\s12_eda_05.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_05`.

## dsts functions used
`read_csv`, `stratified_split`, `DataFrame::select_columns`, `correlation_matrix`, `pearson`, `spearman`, `write_svg_heatmap`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- Pearson measures linear association; Spearman measures monotone association.
- Always note how many complete pairs a coefficient is based on.

## Next module
Continue with **06_target_relationships**.
