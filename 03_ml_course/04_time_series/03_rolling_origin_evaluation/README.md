# Rolling-origin evaluation

## Method and core idea

Random cross-validation folds are wrong for a series: a fold could train on 1960 and validate on
1950. Rolling-origin (expanding-window) evaluation fixes this by choosing origins o = 48, 60, 72,
... inside the training prefix, fitting each candidate on months [0, o) and scoring its one-step
forecasts on [o, o + 12). Averaging the per-origin RMSE selects hyper-parameters using only the
past, while the final 20% of the series stays untouched until the single held-out evaluation. The
held-out months are then forecast one step ahead: predict month t, reveal y_t to the model with
`observe()`, predict t + 1, so the reported error is "one month ahead with the true history", not
a 29-month extrapolation. This module owns `Workflow.hpp`, the pipeline that every time-series
module reuses.

## Dataset and why

AirPassengers: with 115 training months and 12-month validation windows there are enough origins
to see the error drift as the level grows, which is exactly what a single split would hide.

## Prerequisites

`../01_forecasting_baselines/` and `../02_lag_feature_regression/`. Read `theory.md`,
`math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Workflow.hpp` | header (linked via `ml_core`) | The shared pipeline: `read_passengers`, `Forecast` adapter, candidate grids (`KindSpec`), `workflow(root, csv, kind, argc, argv)` with `--quick` and `--model` | - |
| `01_theory.cpp` | `forecast_rolling_theory` | Why random splits scramble a forecasting problem | prints only |
| `02_math_intuition.cpp` | `forecast_rolling_math_intuition` | Origin windows, RMSE and how one-step evaluation reveals y_t only after forecasting it | prints only |
| `03_implementation.cpp` | `forecast_rolling_implementation` | Seasonal naive and the three-alpha lag grid refitted at every origin; RMSE per origin | `results/03_implementation_results/{per_origin_errors.csv, figures/rmse_per_origin.svg}` |
| `04_end_to_end.cpp` | `forecast_rolling_end_to_end` | Full project via `Workflow.hpp` with `LagRegression` | `results/04_end_to_end_results/full/` or `quick/` |
| `predict.cpp` | `forecast_rolling_predict` | Same program compiled for `--model` inference | `results/predict_results/next_forecast.csv` |
| `tests/model_test.cpp` | `forecast_rolling_tests` | Seasonal lookup, lag indexing, trend extrapolation, exact reload | prints only |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target forecast_rolling_end_to_end forecast_rolling_predict
build\03_ml_course\04_time_series\03_rolling_origin_evaluation\forecast_rolling_end_to_end.exe
build\03_ml_course\04_time_series\03_rolling_origin_evaluation\forecast_rolling_end_to_end.exe --quick
build\03_ml_course\04_time_series\03_rolling_origin_evaluation\forecast_rolling_predict.exe --model 03_ml_course\04_time_series\03_rolling_origin_evaluation\results\04_end_to_end_results\full\model\lag_model.txt
```

## Pipeline stages (Workflow.hpp)

1. Parse flags; with `--model <file>` reload the archive, write `next_forecast.csv` and stop.
2. Read the series; the first 80% is training, the rest is the holdout.
3. EDA on the lag-feature view of the training window.
4. Rolling-origin selection of the candidate grid (`validation/candidate_scores.csv`).
5. Final fit on the whole training window plus the seasonal-naive and last-value references.
6. Save the model and reload it; the first held-out forecast must agree to 1e-12.
7. One-step holdout with `observe()` after every month.
8. RMSE, MAE, MASE and the reference RMSEs (`evaluation/metrics.json`), figures, autocorrelation.
9. `feature_schema.json`, `run_manifest.json`, `report.md`, `execution.log`.

## Results layout

`results/04_end_to_end_results/full/` (or `quick/`) contains `eda/`, `validation/`, `model/`
(`lag_model.txt`, `seasonal_model.txt`, `feature_schema.json`), `inference/`, `evaluation/`
(`predictions.csv`, `metrics.json`, `figures/forecast.svg`, `figures/residuals.svg`),
`run_manifest.json`, `report.md`, `execution.log`. `predict` writes
`results/predict_results/next_forecast.csv`.

## Tests

`ctest --preset course -R forecast_rolling` runs `forecast_rolling_numerical` (the shared
`tests/model_test.cpp` fixture: seasonal lookup, lag indexing, trend extrapolation to 106,
exact reload) and `forecast_rolling_workflow` (`--quick` project).

## Key takeaways

- Select on expanding origins, evaluate once on the chronological tail.
- One-step-ahead error with revealed history is a different (easier) quantity than multi-step
  error; say which one you report.
- A shared workflow makes every model comparable: same split, same origins, same metrics.

## Next

`../04_exponential_smoothing/` replaces the regression with explicit level, trend and season
recursions.
