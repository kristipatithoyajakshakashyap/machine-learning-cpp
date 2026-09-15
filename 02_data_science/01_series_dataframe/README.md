# Module 01 — Series and DataFrame

## Purpose
Everything reads and structures data the way `pandas` does, but with zero
dependencies. The whole toolkit lives in `include/dsts/` and is compiled into
the static library `dsts`.

## Prerequisites

The 01_get_to_know track (vectors, classes, `std::optional`, file I/O) and 00_math_foundations.

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_series_basics.cpp` | `s01_01_series_basics` | A `dsts::Series` is one named column of `OptD` (optional double); `mean`, `median`, `variance`, `stddev`, `min`, `max`, `quantile`, `describe`, `head`, `tail`; data: first 10 sepal_length values of the iris flower dataset (real data) | `pandas.Series` | prints only |
| `02_series_ops.cpp` | `s01_02_series_ops` | Elementwise `add/sub/mul/div`; missing-value propagation; comparison masks `gt/ge/lt/le`; data: first 10 penguin measurements from penguins.csv (real data) | `Series + 10`, `Series > x` | prints only |
| `03_dataframe_basics.cpp` | `s01_03_dataframe_basics` | Build a `dsts::DataFrame`; `shape`, `columns`, `head`, `tail`, `info`, `cell`, `series`, `select_rows`, `select_columns`; data: rows 1-6 of iris.csv (real data) | `pandas.DataFrame` | prints only |
| `04_csv_io.cpp` | `s01_04_csv_io` | `read_csv` with automatic column typing (numeric vs string, empty cell = missing); `write_csv` round-trip; error on missing file; data: iris.csv, tips.csv, flights.csv from the data/ folder (real data) | `pandas.read_csv` / `to_csv` | `iris_first3.csv` under `results/04_csv_io_results/` |
| `05_selecting_and_masks.cpp` | `s01_05_selecting` | Boolean masks, `&&` combinations (`both`), equality masks on string columns, index-list subsetting; data: iris.csv and tips.csv (real data) | `df[mask]`, `df[(a) & (b)]`, `df.iloc` | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/01_series_dataframe/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s01_01_series_basics
.\build\02_data_science\01_series_dataframe\s01_01_series_basics.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Series`.

- `01_series_basics.cpp`: `fmt`
- `02_series_ops.cpp`: `fmt`, `to_text`
- `03_dataframe_basics.cpp`: `fmt`
- `04_csv_io.cpp`: `read_csv`, `write_csv`
- `05_selecting_and_masks.cpp`: `read_csv`

## Rules the whole course follows
- A missing numeric cell is `std::nullopt` and **never enters the maths**.
- `fmt()` prints doubles with trailing zeros trimmed; `to_text()` prints `NA`
  for a missing cell.
- `split_csv` is quote-aware: `"Braund, Mr. Owen Harris"` is one cell.
- All output is deterministic: real datasets, fixed seeds, fixed clock.

## Key takeaways
- A `Series` is a named column of `OptD`; missing cells never enter the maths.
- A `DataFrame` is a set of equal-length numeric or string columns addressed by name.
- `read_csv` infers each column type; boolean masks select rows without copying logic.

## Next module

**02_cleaning** - missing values, duplicates, outliers, encoding, scaling and string columns.
