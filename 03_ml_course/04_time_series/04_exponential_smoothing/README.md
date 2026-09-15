# Exponential smoothing (additive Holt-Winters)

## Method and core idea

Exponential smoothing forecasts with a weighted average whose weights decay geometrically into the
past, so recent months matter most. Additive Holt-Winters keeps three smoothed components and
updates them after every observation: the *level* l_t = alpha (y_t - s_{t-m}) + (1 - alpha)
(l_{t-1} + b_{t-1}), the *trend* b_t = beta (l_t - l_{t-1}) + (1 - beta) b_{t-1}, and the seasonal
offset s_t = gamma (y_t - l_t) + (1 - gamma) s_{t-m} for period m = 12. The one-step forecast is
l + b + s for the coming month. The weights alpha, beta, gamma in (0, 1) decide how quickly each
component reacts; the initial state comes from the first two seasons. Because the state is updated
by a recursion, `observe()` is the natural way to follow a live series, and the model is fully
described by three numbers plus the current state, which is what the archive stores.

## Dataset and why

AirPassengers has the textbook shape for Holt-Winters: a steady trend and a repeating 12-month
season. The implementation lesson plots the level, trend and season paths so each component can
be seen doing its job.

## Prerequisites

Modules 01-03 (seasonal baselines, one-step evaluation, rolling-origin selection). Read
`theory.md`, `math_intuition.md` and `implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `Model.hpp` | header (linked via `ml_core`) | `forecast_course::HoltWinters`: public `alpha`, `beta`, `gamma`, `period`, `level`, `trend`, `seasonal`, `observed`; `initialize`, `fit`, `next`, `observe`, `save`/`load` | - |
| `01_theory.cpp` | `forecast_hw_theory` | What the three components mean | prints only |
| `02_math_intuition.cpp` | `forecast_hw_math_intuition` | The recursions traced by hand on eight points with season length four, compared with `fit()` | prints only |
| `03_implementation.cpp` | `forecast_hw_implementation` | Fit on AirPassengers (0.5, 0.1, 0.3) and replay the recursions to record every component | `results/03_implementation_results/{components.csv, figures/level.svg, figures/trend.svg, figures/season.svg}` |
| `04_end_to_end.cpp` | `forecast_hw_end_to_end` | Full project via `Workflow.hpp`: 27-point grid over alpha, beta, gamma in {0.2, 0.5, 0.8} on rolling origins | `results/04_end_to_end_results/full/` or `quick/` |
| `predict.cpp` | `forecast_hw_predict` | Same program compiled for `--model` inference | `results/predict_results/next_forecast.csv` |
| `tests/model_test.cpp` | `forecast_hw_tests` | Beats seasonal naive on a synthetic series, `observe()` matches `fit()`, level recursion, exact reload | prints only |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target forecast_hw_end_to_end forecast_hw_predict
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_end_to_end.exe
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_end_to_end.exe --quick
build\03_ml_course\04_time_series\04_exponential_smoothing\forecast_hw_predict.exe --model 03_ml_course\04_time_series\04_exponential_smoothing\results\04_end_to_end_results\full\model\holt_winters_model.txt
```

The best triple is refitted on the first 80% of the series; the held-out months are forecast one
step ahead with the state updated after every observed month.

## Results layout

`results/04_end_to_end_results/full/` (or `quick/`) contains `eda/`,
`validation/candidate_scores.csv` (RMSE per alpha, beta, gamma and origin),
`model/{holt_winters_model.txt, feature_schema.json}`, `inference/reload_verification.json`,
`evaluation/{predictions.csv, metrics.json, figures/forecast.svg, figures/residuals.svg}`,
`run_manifest.json`, `report.md` and `execution.log`. `predict` writes
`results/predict_results/next_forecast.csv`.

## Tests

`ctest --preset course -R forecast_hw` runs `forecast_hw_numerical` (`tests/model_test.cpp`: on a
96-point trend-plus-season series the hold-out RMSE of Holt-Winters is below seasonal naive;
`fit(all but last)` + `observe(last)` equals `fit(all)` to 1e-9; the level recursion matches the
formula after one surprise observation; save/load reproduces `next()`, `observed` and `alpha`)
and `forecast_hw_workflow` (`--quick` project).

## Key takeaways

- Three interpretable components; read `components.csv` to see what the model believes.
- Higher weights react faster but follow noise; the rolling-origin grid decides.
- The state is the model: `observe()` and the archive both carry level, trend and season.

## Next

`../05_arima/` models the differenced series with autoregressive and moving-average terms.
