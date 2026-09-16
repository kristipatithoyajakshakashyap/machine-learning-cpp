# Forecasting with C++17

## What this track teaches

Time-series forecasting predicts the next value of an ordered sequence from its own past. Order
changes everything: rows never get shuffled, validation must only ever train on the past, and a
model must be judged against trivial references before it is believed. Five modules build up one
consistent toolkit: seasonal-naive baselines, lag-feature regression (turn the series into a
supervised table), rolling-origin evaluation (expanding-window model selection), additive
Holt-Winters exponential smoothing (level, trend, season recursions) and a simplified
ARIMA (difference, autoregress, moving average). Every model implements the same tiny
interface: `fit(series)`, `next()` (one-step forecast), `observe(actual)` (advance the state
without refitting), `save`/`load`.

## Dataset and why

All modules use AirPassengers (`helper/data/air_passengers.csv`): 144 monthly totals of
international airline passengers, 1949-1960, in thousands, the classic Box and Jenkins series
(https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html). It has a clear
upward trend and a strong yearly season, so each method's strengths and blind spots show up
immediately, and it is short enough that every lesson runs in well under a second.

## Prerequisites

C++ vectors and lambdas, least-squares regression (`../01_supervised/01_regression/`), RMSE/MAE,
and the `helper/` toolbox (`Artifacts`, `Plot`, `archive`).

## Modules and shared files

| Path | Target prefix | Model | Selection grid |
|---|---|---|---|
| `01_forecasting_baselines/` | `forecast_baselines_` | `SeasonalNaive` (`Model.hpp`) | none (reference only) |
| `02_lag_feature_regression/` | `forecast_lag_` | `LagRegression` ridge on lags 1, 2, 12 + seasonal sin/cos (`Model.hpp`) | alpha in {0, 0.1, 10} |
| `03_rolling_origin_evaluation/` | `forecast_rolling_` | the shared pipeline itself (`Workflow.hpp`), run with `LagRegression` | alpha in {0, 0.1, 10} |
| `04_exponential_smoothing/` | `forecast_hw_` | `HoltWinters` additive level/trend/season (`Model.hpp`) | alpha, beta, gamma in {0.2, 0.5, 0.8}^3 |
| `05_arima/` | `forecast_arima_` | `Arima(p, d, q)` (`Model.hpp`) | p, q in {1, 2}, d = 1 |

`03_rolling_origin_evaluation/Workflow.hpp` is the one end-to-end pipeline every module's
`04_end_to_end.cpp` calls: read series, EDA on the lag-feature view, rolling-origin selection
(origins 48, 60, ... on the training prefix, each validated on the next 12 months by RMSE), final
fit on the first 80%, save + reload check, one-step held-out evaluation of the last 20% with the
state updated after every observed month, RMSE/MAE/MASE against the seasonal-naive and last-value
references, autocorrelation, manifest and report. It also implements the `--model` inference path.

## Build and run

From the repository root with `D:\msys64\ucrt64\bin` on `PATH`:

```powershell
cmake --preset course
cmake --build --preset course --target forecast_hw_end_to_end forecast_hw_predict forecast_hw_tests
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_end_to_end.exe          # full run
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_end_to_end.exe --quick  # same, writes under quick/
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_predict.exe --model 03_ml_course\04_time_series\04_exponential_smoothing\results\04_end_to_end_results\full\model\holt_winters_model.txt
```

Replace `forecast_hw` by any prefix. The model file is `seasonal_model.txt` (baselines),
`lag_model.txt` (lag regression and rolling origin), `holt_winters_model.txt` or
`arima_model.txt`. `--quick` uses the same data and writes the identical artifact set under
`quick/` so CTest never overwrites a `full/` run. `predict --model <file>` reloads the archive
(which contains the observed history and every fitted transformation) and writes one next-month
forecast to `results/predict_results/next_forecast.csv`.

## Results layout

Every executable owns `results/<cpp-stem>_results/` inside its module. The project writes
`results/04_end_to_end_results/full/` (or `quick/`) with `eda/` (lag-feature EDA and
`autocorrelation.csv`), `validation/candidate_scores.csv` (RMSE per candidate and origin),
`model/<kind>_model.txt` plus `feature_schema.json` (modules 01-03 also save `lag_model.txt` and
`seasonal_model.txt` as references), `inference/reload_verification.json`,
`evaluation/{predictions.csv, metrics.json, figures/forecast.svg, figures/residuals.svg}`,
`run_manifest.json`, `report.md` and `execution.log`. Forecasts are one month ahead with the true
history revealed after each step. No multi-step claim is made.

## Tests

```powershell
ctest --preset course -R "forecast_.*_numerical"   # tests/model_test.cpp fixtures
ctest --preset course -R "forecast_.*_workflow"    # each 04_end_to_end --quick
```

## Key takeaways

- Split chronologically and validate on rolling origins. Random folds leak the future.
- Always report the seasonal-naive and last-value references next to the model.
- `observe()` lets you track a live series with a fitted model without refitting. The archive must
  carry the history for this to work after a reload.

## Next

`../05_text_classification/` leaves numeric series for token counts and TF-IDF.