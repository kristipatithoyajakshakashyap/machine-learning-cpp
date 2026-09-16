# Module 05: Joins and reshaping

## Purpose
Relational joins and long/wide reshaping on top of the DataFrame. The helpers:
`include/dsts/join.hpp`.

## Prerequisites

Module 01_series_dataframe (DataFrame columns and row selection).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_merge_join.cpp` | `s05_01_merge` | Left join attaching a day-level lookup table to every tip. Data: tips.csv (real data, 244 restaurant tips) | `pd.merge(...)` | prints only |
| `02_inner_vs_left.cpp` | `s05_02_inner` | Inner vs left semantics and NA fill for unmatched rows. Data: titanic.csv (real data, 891 passengers) | `pd.merge(how=...)` | prints only |
| `03_concat.cpp` | `s05_03_concat` | Vertical stacking of frames with identical columns. Data: iris.csv (real data, 150 flowers) | `pd.concat(axis=0)` | prints only |
| `04_pivot.cpp` | `s05_04_pivot` | Mean grid: index × columns of a numeric value. Data: penguins.csv (real data, 344 penguins) and tips.csv (244 tips) | `df.pivot_table(...)` | prints only |
| `05_melt.cpp` | `s05_05_melt` | Long form: variable/value pairs from wide columns. Data: iris.csv (real data, 150 flowers) | `df.melt()` | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/05_joins_reshape/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s05_01_merge
.\build\02_data_science\05_joins_reshape\s05_01_merge.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`.

- `01_merge_join.cpp`: `fmt`, `merge`, `read_csv`
- `02_inner_vs_left.cpp`: `merge`, `read_csv`
- `03_concat.cpp`: `concat_rows`, `fmt`, `read_csv`
- `04_pivot.cpp`: `pivot_table`, `read_csv`
- `05_melt.cpp`: `fmt`, `melt`, `read_csv`

## Key ideas
- Merge keys must line up. In `merge()` the right frame's key values must be
  unique, exactly like a primary key in a database. Key order does not matter.
  The match is looked up, not assumed.
- Inner vs left. Inner joins answer which rows exist on both sides. Left joins
  answer everything on the left, enriched where possible, with NA wherever
  there is no match.
- Pivot and melt are inverses. Wide grids are easy to read. Long rows are
  easy to group and plot. Most plotting helpers prefer long form.

## Key takeaways
- `merge()` looks keys up. The right-hand key must be unique like a database primary key.
- Inner keeps matches only. Left keeps every left row and fills NA where nothing matches.
- `pivot` and `melt` are inverses: wide for reading, long for grouping and plotting.

## Next module

06_feature_engineering - derived columns, binning, log transforms, indicator features.