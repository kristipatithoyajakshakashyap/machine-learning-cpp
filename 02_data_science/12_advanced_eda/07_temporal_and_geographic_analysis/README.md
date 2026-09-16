# Temporal and geographic analysis

## Purpose
Switches to the AirPassengers series to show chronological diagnostics: lag-12 differences and ratios reveal trend and growing seasonal amplitude. No geographic dataset is fabricated. The lesson explains when spatial analysis applies and how it must be validated.

## Prerequisites
Modules 03_eda, 07_statistics_tests and 08_train_test. All Titanic analysis
uses only the seeded 80% training partition (`dsts::stratified_split` on
`Survived`, seed 42) so that nothing is learned from the hold-out rows.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_analysis.cpp` | `s12_eda_07` | lag-12 differences and ratios, seasonality, why time series must not be shuffled | `temporal_diagnostics.csv`, `passengers.svg`, `temporal.md` under `results/01_analysis_results/` |

`01_analysis.cpp` is a one-line driver. The worked implementation is the
corresponding stage function in `../eda_workflow.hpp`, which also documents
every output file.

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s12_eda_07
.\build\02_data_science\12_advanced_eda\07_temporal_and_geographic_analysis\s12_eda_07.exe
```

The lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes to
`results/01_analysis_results/` next to its source (injected as
`RUN_OUTPUT_DIR`). It is registered as ctest `s12_eda_07`.

## dsts functions used
`read_csv`, `DataFrame::numeric`, `DataFrame::strings`, `write_svg_line` (Titanic split still runs for consistency).

## Exercise
Rerun with a different training seed, compare effect sizes and sample counts,
and identify which conclusions remain stable. Explain why using a held-out
outcome to select preprocessing would invalidate later evaluation.

## Key takeaways
- Preserve chronology: never randomly split forecasting data.
- Growing seasonal amplitude suggests ratios or log differences.

## Next module
Continue with 08_eda_report.