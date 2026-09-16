# Module 09: Time series

## Purpose
Ordered data needs ordered thinking: trend, seasonality, and the baseline
forecasts that every real model has to beat. Toolkit added this module:
`include/dsts/timeseries.hpp`. It provides `rolling_mean()`, `autocorr()`,
`diff()` and `rmse()`. Every number was checked against pandas/numpy.scipy.

## Prerequisites

Module 03_eda (correlation) and 06_feature_engineering (log transform).

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_components.cpp` | `s09_01_components` | Series + 12-month moving average. Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960) | `s.rolling(12).mean()` | `m09_01_passengers.svg` under `results/01_components_results/` |
| `02_seasonality.cpp` | `s09_02_seasonality` | Month-of-year profile vs overall. Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960) | `s.groupby(month).mean()` | prints only |
| `03_forecast_benchmark.cpp` | `s09_03_bench` | Naive vs mean vs seasonal-naive RMSE. Data: air_passengers.csv (real data, 144 monthly totals) | `shift()` + `mean_squared_error` | prints only |
| `04_trend_linreg.cpp` | `s09_04_trend` | Least-squares trend, R2 = corr^2. Data: air_passengers.csv (real data, 144 monthly totals) | `numpy.polyfit(t, y, 1)` | prints only |
| `05_acf.cpp` | `s09_05_acf` | Autocorrelation across lags. Data: air_passengers.csv (real data, 144 monthly totals) | `Series.autocorr(lag)` | prints only |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/09_time_series/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s09_01_components
.\build\02_data_science\09_time_series\s09_01_components.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::DataFrame`, `dsts::OptD`, `dsts::Series`.

- `01_components.cpp`: `fmt`, `read_csv`, `rolling_mean`, `write_svg_line`
- `02_seasonality.cpp`: `fmt`, `read_csv`
- `03_forecast_benchmark.cpp`: `fmt`, `read_csv`, `rmse`
- `04_trend_linreg.cpp`: `fmt`, `pearson`, `read_csv`
- `05_acf.cpp`: `autocorr`, `fmt`, `read_csv`

## Key ideas
- Trend vs noise. Passengers more than tripled over 12 years (112 -> 417) while
  the 12-month average rises smoothly from ~126 to ~476.
- Seasonality is a 12-month fingerprint. July/August run ~71 above the
  yearly mean, November ~47 below.
- Always benchmark. The seasonal naive (49.99 RMSE) crushes the flat
  mean (219.44) and the plain naive (137.33) on the held-out 24 months.
- The ACF says it in one glance. Autocorr peaks at lag 12 (0.991),
  proving the annual cycle dominates.

## Key takeaways
- Decompose a series into trend, seasonality and noise before modelling it.
- Always benchmark against naive, mean and seasonal-naive forecasts on held-out months.
- The autocorrelation function reveals the dominant cycle in one glance.

## Next module

10_dimensionality - PCA: explained variance, projection, reconstruction, loadings.