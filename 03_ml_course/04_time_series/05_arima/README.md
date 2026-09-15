# ARIMA

## Method and core idea

ARIMA(p, d, q) models the d-th difference of a series as a linear function of its own p most
recent values (the autoregressive part) and the q most recent forecast errors (the moving-average
part): w_t = c + phi_1 w_{t-1} + ... + phi_p w_{t-p} + theta_1 e_{t-1} + ... + theta_q e_{t-q}
+ e_t. Differencing removes trend so the remaining series is roughly stationary; the sample
autocorrelation before and after differencing shows why it is needed. The forecast of the next
difference is integrated back by adding the last observed levels. This module's estimator is
deliberately simple: AR coefficients by ordinary least squares on the differenced series, then MA
coefficients by conditional sum of squares with plain gradient descent, holding the AR part fixed.
`observe()` records the new residual (needed by the MA terms) and extends the history without
refitting. See `theory.md` for what a full maximum-likelihood ARIMA adds.

## Dataset and why

AirPassengers: one difference removes the trend, the differenced ACF then peaks at lag 12, and a
small (p, q) grid is enough to show how the terms trade off. The math-intuition lesson uses an
eight-point series whose differences alternate 3, 4, 3, 4 so phi is visibly negative.

## Prerequisites

Modules 01-04, in particular autocorrelation from the rolling-origin EDA and the state-update
habit of Holt-Winters; least squares. Read `theory.md`, `math_intuition.md` and
`implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Model.hpp` | header (linked via `ml_core`) | `forecast_course::Arima`: public `p`, `d`, `q`, `constant`, `phi`, `theta`, `history`, `residuals`; static `difference`, `fit`, `next`, `observe`, `save`/`load` | - |
| `01_theory.cpp` | `forecast_arima_theory` | What p, d and q mean | prints only |
| `02_math_intuition.cpp` | `forecast_arima_math_intuition` | Difference an eight-point series, fit AR(1) by hand, integrate the forecast back and compare with `next()` | prints only |
| `03_implementation.cpp` | `forecast_arima_implementation` | ARIMA(1,1,1) on AirPassengers: differenced series, ACF before/after differencing (24 lags), residuals | `results/03_implementation_results/{differences.csv, autocorrelation.csv, figures/differences.svg, figures/autocorrelation.svg, figures/residuals.svg}` |
| `04_end_to_end.cpp` | `forecast_arima_end_to_end` | Full project via `Workflow.hpp`: (p, q) in {1, 2} x {1, 2}, d = 1, selected on rolling origins | `results/04_end_to_end_results/full/` or `quick/` |
| `predict.cpp` | `forecast_arima_predict` | Same program compiled for `--model` inference | `results/predict_results/next_forecast.csv` |
| `tests/model_test.cpp` | `forecast_arima_tests` | AR(1) recovery from seeded noise, differencing round trip, exact reload | prints only |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target forecast_arima_end_to_end forecast_arima_predict
build\03_ml_course\04_time_series\05_arima\forecast_arima_end_to_end.exe
build\03_ml_course\04_time_series\05_arima\forecast_arima_end_to_end.exe --quick
build\03_ml_course\04_time_series\05_arima\forecast_arima_predict.exe --model 03_ml_course\04_time_series\05_arima\results\04_end_to_end_results\full\model\arima_model.txt
```

The best order is refitted on the first 80% of the series; the held-out months are forecast one
step ahead while the residual history is extended after every observed month.

## Results layout

`results/04_end_to_end_results/full/` (or `quick/`) contains `eda/`,
`validation/candidate_scores.csv` (RMSE per p, d, q and origin), `model/{arima_model.txt,
feature_schema.json}` (orders, constant, phi, theta, history and residuals),
`inference/reload_verification.json`, `evaluation/{predictions.csv, metrics.json,
figures/forecast.svg, figures/residuals.svg}`, `run_manifest.json`, `report.md` and
`execution.log`. `predict` writes `results/predict_results/next_forecast.csv`.

## Tests

`ctest --preset course -R forecast_arima` runs `forecast_arima_numerical` (`tests/model_test.cpp`:
OLS recovers phi = 0.7 within 0.1 from 500 seeded AR(1) points and allocates no theta for q = 0;
the second difference of t^2 is the constant 2 with two fewer elements; ARIMA(0,2,0) on squares
forecasts 121 and, after observing it, 144 with a zero residual; save/load reproduces orders,
theta, residuals and `next()` exactly) and `forecast_arima_workflow` (`--quick` project).

## Key takeaways

- Difference until the ACF stops decaying slowly, then model what is left.
- AR terms use past values, MA terms use past errors; the errors must be stored to forecast.
- A simple estimator is enough to learn the mechanics; production ARIMA uses full likelihood.

## Next

`../../05_text_classification/` leaves numeric series for token counts and TF-IDF.
