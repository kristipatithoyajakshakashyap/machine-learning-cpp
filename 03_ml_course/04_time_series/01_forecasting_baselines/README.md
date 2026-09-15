# Forecasting baselines

## Method and core idea

A forecast is only useful if it beats something trivial, so this module builds the references
every later model is measured against. The *naive* forecast repeats the last observed value; the
*expanding mean* averages all history and ignores dynamics; the *seasonal naive* forecast repeats
the value one season ago (yhat_t = y_{t-12} for monthly data), which already captures the yearly
cycle of AirPassengers. `SeasonalNaive` keeps the whole history so `observe()` can extend it and
`next()` always looks back exactly one period; with period 1 it degenerates to the naive forecast.
The module also introduces MASE, the mean absolute error divided by the in-sample seasonal-naive
MAE, so that "1.0" always means "no better than repeating last year".

## Dataset and why

AirPassengers (144 monthly totals, 1949-1960): trend plus a strong season, so the three baselines
differ visibly and the seasonal one is hard to beat. See `../README.md`.

## Prerequisites

C++ vectors, RMSE/MAE, the idea of time-ordered data. Read `theory.md`, `math_intuition.md` and
`implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Model.hpp` | header (linked via `ml_core`) | `forecast_course::SeasonalNaive`: `fit`, `next`, `observe`, `save`/`load` | - |
| `01_theory.cpp` | `forecast_baselines_theory` | Why forecasts must beat simple references | prints only |
| `02_math_intuition.cpp` | `forecast_baselines_math_intuition` | yhat_t = y_{t-12} worked on a number, MASE definition | prints only |
| `03_implementation.cpp` | `forecast_baselines_implementation` | Three-point sanity check, then last 20% forecast one step ahead by naive, seasonal naive and expanding mean with RMSE/MAE | `results/03_implementation_results/{forecasts.csv, rmse_table.csv, figures/baselines.svg}` |
| `04_end_to_end.cpp` | `forecast_baselines_end_to_end` | Full project via `Workflow.hpp` with the seasonal-naive family | `results/04_end_to_end_results/full/` or `quick/` |
| `predict.cpp` | `forecast_baselines_predict` | Same program compiled for `--model` inference | `results/predict_results/next_forecast.csv` |
| `tests/model_test.cpp` | `forecast_baselines_tests` | Seasonal lookup, `observe()` history, lag indexing, trend extrapolation, exact reload | prints only |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target forecast_baselines_end_to_end forecast_baselines_predict
build\03_ml_course\04_time_series\01_forecasting_baselines\forecast_baselines_end_to_end.exe
build\03_ml_course\04_time_series\01_forecasting_baselines\forecast_baselines_end_to_end.exe --quick
build\03_ml_course\04_time_series\01_forecasting_baselines\forecast_baselines_predict.exe --model 03_ml_course\04_time_series\01_forecasting_baselines\results\04_end_to_end_results\full\model\seasonal_model.txt
```

`--quick` writes the same artifacts under `quick/` instead of `full/`. The `--model` command
reloads the archive (period plus observed history) and writes one next-month forecast.

## Results layout

`results/04_end_to_end_results/full/` (or `quick/`) contains `eda/` (with
`autocorrelation.csv`), `validation/candidate_scores.csv`, `model/{seasonal_model.txt,
lag_model.txt, feature_schema.json}` (this module also saves the tuned lag model as a
comparison), `inference/reload_verification.json`, `evaluation/{predictions.csv, metrics.json,
figures/forecast.svg, figures/residuals.svg}`, `run_manifest.json`, `report.md` and
`execution.log`. `predict` writes `results/predict_results/next_forecast.csv`.

## Tests

`ctest --preset course -R forecast_baselines` runs `forecast_baselines_numerical`
(`tests/model_test.cpp`: `SeasonalNaive(3)` fitted on {1,2,3} forecasts 1, then 2 after
observing 4; lag features index y[t-1] and y[t-12]; an unregularised lag regression extrapolates a
line to 106; save/load reproduces the forecast) and `forecast_baselines_workflow` (the project
with `--quick`).

## Key takeaways

- Seasonal naive is the reference that matters on seasonal data; report it next to every model.
- MASE makes errors comparable across series and models.
- `observe()` is how a fitted forecaster follows a live series without refitting.

## Next

`../02_lag_feature_regression/` turns the series into a supervised table and fits a regression.
