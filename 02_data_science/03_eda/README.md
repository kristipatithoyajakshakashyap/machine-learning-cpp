# Module 03: Exploratory data analysis

## Purpose
EDA turns raw columns into numbers you reason with. The helpers:
`include/dsts/eda.hpp`.

## Prerequisites

Modules 01 and 02 (loading and cleaning a DataFrame).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_summary_and_counts.cpp` | `s03_01_summary` | Multi-column `summary()` block and `value_counts()` frequency tables. Data: iris.csv and tips.csv (real data) | `df.describe()`, `df.value_counts()` | prints only |
| `02_distributions.cpp` | `s03_02_distrib` | Equal-width histograms, proportions, and right-skewed tip data. Data: iris.csv and tips.csv (real data) | `numpy.histogram`, `plt.hist` | prints only |
| `03_correlation.cpp` | `s03_03_correlation` | Pearson and Spearman, and the whole-matrix `correlation_matrix()`. Data: iris.csv and tips.csv (real data) | `df.corr()`, `scipy.stats.pearsonr/spearmanr` | prints only |
| `04_grouped_analysis.cpp` | `s03_04_grouped` | count / mean / sd inside each group. Data: tips.csv and iris.csv (real data) | `df.groupby(...).mean()` | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/03_eda/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s03_01_summary
.\build\02_data_science\03_eda\s03_01_summary.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::Histogram`, `dsts::Series`.

- `01_summary_and_counts.cpp`: `read_csv`, `summary`, `value_counts`
- `02_distributions.cpp`: `fmt`, `histogram`, `read_csv`
- `03_correlation.cpp`: `correlation_matrix`, `fmt`, `pearson`, `read_csv`, `spearman`
- `04_grouped_analysis.cpp`: `fmt`, `group_mean`, `read_csv`

## Key ideas
- Correlation does not equal causation. The big-tip story is exactly the sort
  of claim the matrix alone does not prove (lesson 3).
- Histograms are the fastest way to see shape, center, spread and tails.
- Grouped means are a one-line way to compare categories before any
  statistical test.

## Key takeaways
- `summary()` and `value_counts()` are the first two things to run on any table.
- Histograms show shape, centre, spread and tails faster than any statistic.
- Pearson measures linear relation, Spearman monotone relation. Neither proves causation.

## Next module

04_plots - line, scatter, bar, histogram, box and heatmap charts as SVG.