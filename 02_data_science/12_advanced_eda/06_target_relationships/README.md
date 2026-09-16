# Target relationships

## Purpose
Computes the survival rate by Sex with Wilson 95% score intervals, showing how sampling uncertainty attaches to a group rate and why these are training associations rather than causal effects.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_06` | group means of a binary target, Wilson score interval | `survival_by_sex.csv`, `target.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver. The worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_06
.\build\02_data_science\12_advanced_eda\06_target_relationships\s12_eda_06.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_06`.

## dsts functions used
`read_csv`, `stratified_split`, `group_mean`.

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- A rate without an interval hides its sample size.
- Family dependence between passengers makes intervals too narrow.

## Next module
Continue with 07_temporal_and_geographic_analysis.