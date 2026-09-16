# Lag-feature regression

## Method and core idea

Forecasting becomes ordinary regression once each month t is described by features that were
already known before t: the previous value (lag 1), the one before (lag 2), the same month last
year (lag 12), the time index for the trend, and a sine/cosine pair of the month position so
December and January sit next to each other. `LagRegression` standardizes those six features and
fits a ridge regression (`ElasticNet` with l1 ratio 0 from the supervised track). The L2 penalty
`alpha` keeps the correlated lags from producing unstable coefficients. After fitting, the
coefficients and the scaler are frozen and `observe()` only appends to the history, so every
held-out forecast uses fresh lag values with the same model. Nothing at or after t enters row t.
That leakage rule is the whole difficulty of the method.

## Dataset and why

AirPassengers: lag 12 captures the season, the time index the trend. Building the lag features
needs the 12 warm-up months, fitting needs two years (24 points) of history, and the last 20 %
(29 months) is the one-step holdout. See `../README.md`.

## Prerequisites

`../01_forecasting_baselines/` and ridge regression
(`../../01_supervised/01_regression/`). Read `theory.md`, `math_intuition.md` and
`implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Model.hpp` | header (linked via `ml_core`) | `features(history, t)` and `forecast_course::LagRegression`: `fit`, `next`, `observe`, `save`/`load` | - |
| `01_theory.cpp` | `forecast_lag_theory` | The leakage rule and which lags are allowed | prints only |
| `02_math_intuition.cpp` | `forecast_lag_math_intuition` | The feature vector, the penalised loss and why sin/cos encode the season | prints only |
| `03_implementation.cpp` | `forecast_lag_implementation` | Baseline sanity check, the lag table, ridge fit (alpha 0.1) on the first 80%, one-step forecasts for the rest | `results/03_implementation_results/{lag_features.csv, coefficients.csv, fitted_vs_actual.csv, figures/fitted_vs_actual.svg}` |
| `04_end_to_end.cpp` | `forecast_lag_end_to_end` | Full project via `Workflow.hpp` with alpha in {0, 0.1, 10} selected on rolling origins | `results/04_end_to_end_results/full/` or `quick/` |
| `predict.cpp` | `forecast_lag_predict` | Same program compiled for `--model` inference | `results/predict_results/next_forecast.csv` |
| `tests/model_test.cpp` | `forecast_lag_tests` | Seasonal lookup, lag indexing, trend extrapolation, exact reload | prints only |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target forecast_lag_end_to_end forecast_lag_predict
build\03_ml_course\04_time_series\02_lag_feature_regression\forecast_lag_end_to_end.exe
build\03_ml_course\04_time_series\02_lag_feature_regression\forecast_lag_end_to_end.exe --quick
build\03_ml_course\04_time_series\02_lag_feature_regression\forecast_lag_predict.exe --model 03_ml_course\04_time_series\02_lag_feature_regression\results\04_end_to_end_results\full\model\lag_model.txt
```

The archive stores the regression, the fitted scaler and the observed history, so `predict`
rebuilds the next feature row itself.

## Results layout

`results/04_end_to_end_results/full/` (or `quick/`) contains `eda/` (lag-feature statistics and
`autocorrelation.csv`), `validation/candidate_scores.csv` (RMSE per alpha and origin),
`model/{lag_model.txt, seasonal_model.txt, feature_schema.json}`,
`inference/reload_verification.json`, `evaluation/{predictions.csv, metrics.json,
figures/forecast.svg, figures/residuals.svg}`, `run_manifest.json`, `report.md` and
`execution.log`. `predict` writes `results/predict_results/next_forecast.csv`.

## Tests

`ctest --preset course -R forecast_lag` runs `forecast_lag_numerical` (`tests/model_test.cpp`:
`features(y, 24)[0] == y[23]` and `[2] == y[12]`. Alpha = 0 on y_t = 2t + 10 forecasts 106 within
0.1. Save/load reproduces `next()` to 1e-12, plus the seasonal-naive checks) and
`forecast_lag_workflow` (`--quick` project).

## Key takeaways

- Build features only from the past. The time index and lag 12 carry trend and season.
- Regularise: lags are highly correlated and an unpenalised fit is fragile.
- Freeze coefficients at fit time and let `observe()` supply new lags. That is honest one-step
  evaluation.

## Next

`../03_rolling_origin_evaluation/` chooses alpha without touching the holdout.