# Module 06: Feature engineering

## Purpose
Turning raw columns into better inputs for analysis and models. The helpers:
`include/dsts/features.hpp`.

## Prerequisites

Modules 02_cleaning and 03_eda (scaling, correlation).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_derived_columns.cpp` | `s06_01_derived` | Ratios/percentages from elementwise Series arithmetic. Data: tips.csv (real data, 244 restaurant tips) | `df.a / df.b` | prints only |
| `02_binning.cpp` | `s06_02_binning` | `cut()` with explicit edges and `qcut_edges()` equal-count bins. Data: titanic.csv (real data) and tips.csv (real data) | `pd.cut`, `pd.qcut` | prints only |
| `03_log_transform.cpp` | `s06_03_log` | `log1p()` linearises exponential growth. Data: flights.csv (real data, 144 monthly values) | `numpy.log1p` | prints only |
| `04_indicator_features.cpp` | `s06_04_indicators` | Presence, child and family-size flags vs survival. Data: titanic.csv (real data, 891 passengers) | `.notna().astype(int)` | prints only |
| `05_feature_correlation.cpp` | `s06_05_corr` | Rank features by correlation with the target. Data: titanic.csv (real data, 891 passengers) | `df.corr()["Survived"]` | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/06_feature_engineering/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s06_01_derived
.\build\02_data_science\06_feature_engineering\s06_01_derived.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Series`.

- `01_derived_columns.cpp`: `fmt`, `group_mean`, `read_csv`
- `02_binning.cpp`: `cut`, `fmt`, `qcut_edges`, `read_csv`, `value_counts`
- `03_log_transform.cpp`: `fmt`, `log1p`, `pearson`, `read_csv`
- `04_indicator_features.cpp`: `fmt`, `group_mean`, `read_csv`
- `05_feature_correlation.cpp`: `fmt`, `pearson`, `read_csv`

## Key ideas
- Binning turns scale into buckets. Ages into child/teen/adult/senior are
  easier to reason about than raw years. Wrong edges give the wrong story.
  Choose them deliberately.
- Log transforms tame right tails and make multiplicative growth additive.
  In lesson 3, correlation with time rises to ~0.95 after `log1p`.
- Every engineered feature must be checked. `has_cabin` alone separates
  survivors (0.67 vs 0.30), showing why wealth/booking signals matter.

## Key takeaways
- New features come from arithmetic, binning and transforms of existing columns.
- `log1p` tames right tails and turns multiplicative growth into additive growth.
- Check every engineered feature against the target before trusting it.

## Next module

07_statistics_tests - confidence intervals, t/chi-square/KS tests, bootstrap, ANOVA, Bayes.