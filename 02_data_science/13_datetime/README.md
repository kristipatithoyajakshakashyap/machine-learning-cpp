# Module 13: Dates and time indexing

## Purpose
Calendar arithmetic without a library: `include/dsts/datetime.hpp` provides
`Date`, `parse_date` ("YYYY-MM-DD" or "YYYY-MM"), `days_from_civil`,
`weekday` (0 = Sunday), `is_leap_year`, `add_months` (day clamped),
`month_key` ("YYYY-MM") and `quarter_key` ("YYYY-Qn"). All three lessons use
air_passengers.csv (144 monthly totals, 1949-1960).

## Prerequisites

Modules 01_series_dataframe and 09_time_series.

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_parse_and_components.cpp` | `s13_01_parse` | parse the Month column, year/month/weekday, leap years, rows per year. Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960) | `pd.to_datetime`, `.dt.year`, `.dt.dayofweek` | prints only |
| `02_resample.cpp` | `s13_02_resample` | monthly -> quarterly and yearly sums/means, `resampled.csv`. Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960) | `df.resample("Q").sum()` | `resampled.csv` under `results/02_resample_results/` |
| `03_rolling_windows.cpp` | `s13_03_rolling` | rolling mean/std/min/max (window 12), cumulative sum, rank, `rolling.csv` + SVG. Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960) | `.rolling(12)`, `.cumsum()`, `.rank()` | `rolling.csv`, `rolling_mean.svg` under `results/03_rolling_windows_results/` |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/13_datetime/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s13_01_parse
.\build\02_data_science\13_datetime\s13_01_parse.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::Date`, `dsts::OptD`, `dsts::Series`.

- `01_parse_and_components.cpp`: `add_months`, `days_from_civil`, `is_leap_year`, `parse_date`, `quarter_key`, `read_csv`, `to_string`, `weekday`, `weekday_name`
- `02_resample.cpp`: `fmt`, `parse_date`, `read_csv`, `write_csv`
- `03_rolling_windows.cpp`: `fmt`, `parse_date`, `read_csv`, `to_text`, `write_csv`, `write_svg_line`

## Key ideas
- Dates are integers in disguise. `days_from_civil` turns a date into a
  day count, so differences, weekdays and sorting become plain arithmetic.
- Resampling = group by a coarser key. The quarter key `"1949-Q1"` is a
  group label. Sum or mean within each label.
- Rolling windows need a full window. The first `window-1` cells are
  missing, exactly as pandas reports NaN.

## Key takeaways
- A date is a day count in disguise. Differences, weekdays and sorting become arithmetic.
- Resampling is a group-by on a coarser key such as `"1949-Q1"`.
- Rolling windows are missing until the window is full, exactly as pandas reports NaN.

## Next module

You have completed the data-science track. Continue with 03_ml_course.