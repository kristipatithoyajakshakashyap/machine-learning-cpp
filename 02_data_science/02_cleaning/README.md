# Module 02: Data cleaning

## Purpose
Cleaning is where projects spend most of their time. Every helper lives in
`include/dsts/cleaning.hpp`.

## Prerequisites

Module 01_series_dataframe (Series, DataFrame, `read_csv`, masks).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_missing_values.cpp` | `s02_01_missing_values` | Counting missing cells per column, `dropna`, and filling `Age` with its median. Data: penguins.csv and titanic.csv (real data) | `df.dropna()`, `df.fillna()` | prints only |
| `02_duplicates.cpp` | `s02_02_duplicates` | `duplicated()` flags any row equal to an earlier row. `drop_duplicates` keeps the first copy. Data: tips.csv (real data, containing exactly one duplicated row) | `df.duplicated()`, `df.drop_duplicates()` | prints only |
| `03_outliers.cpp` | `s02_03_outliers` | Tukey IQR fences `[q1-1.5·iqr, q3+1.5·iqr]`, outlier flags, and the effect on the mean. Data: tips.csv (real data, 9 total_bill and 9 tip outliers) | box-plot whiskers | prints only |
| `04_encoding.cpp` | `s02_04_encoding` | One-hot encoding one string column into `k` 0/1 indicator columns. Data: tips.csv and iris.csv (real data) | `pandas.get_dummies` | prints only |
| `05_scaling.cpp` | `s02_05_scaling` | Min-max `[0,1]` and z-score (mean 0, sd 1) transformations. Data: tips.csv (real data) | sklearn `MinMaxScaler`, `StandardScaler` | prints only |
| `06_string_columns.cpp` | `s02_06_strings` | Regex-split Titanic `Name` into surname + title, trim/lower-case, and collapse rare titles into Mr/Miss/Mrs/Master/Rare. Writes `titles.csv`. Data: titanic.csv (891 passengers, every row has a Name) | `df["Name"].str.extract(...)`, `value_counts()` | `titles.csv` under `results/06_string_columns_results/` |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/02_cleaning/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s02_01_missing_values
.\build\02_data_science\02_cleaning\s02_01_missing_values.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Series`.

- `01_missing_values.cpp`: `dropna`, `fmt`, `read_csv`
- `02_duplicates.cpp`: `drop_duplicates`, `duplicated`, `read_csv`
- `03_outliers.cpp`: `fmt`, `iqr_fences`, `outlier_mask`, `read_csv`, `to_text`
- `04_encoding.cpp`: `one_hot`, `read_csv`
- `05_scaling.cpp`: `fmt`, `minmax_scale`, `read_csv`, `to_text`, `zscore_scale`
- `06_string_columns.cpp`: `read_csv`, `write_csv`

## Why cleaning matters
- Missing cells are never silently treated as 0. The toolkit keeps them
  as `std::nullopt` and every statistic ignores them.
- Duplicates double-count rows and inflate frequencies.
- Outliers are only removed after inspection, and never blindly.
- Encoding and scaling convert messy human data into the flat numeric
  form a model consumes.
- String columns hide categories. 17 raw titles in `Name` become five
  clean groups (Mr 517, Miss 185, Mrs 126, Master 40, Rare 23).

## Key takeaways
- Count and inspect missing cells before dropping or filling them. Fill from statistics you justify.
- Duplicates and outliers are flagged first and removed only after a human decision.
- One-hot encoding and scaling turn human data into the flat numeric form models consume.

## Next module

03_eda - summaries, histograms, correlations and grouped statistics.