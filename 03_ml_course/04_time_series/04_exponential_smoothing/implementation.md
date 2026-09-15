# Implementation

`HoltWinters` (Model.hpp) stores alpha, beta, gamma, the season length, the current level and trend, one seasonal offset per season position and `observed`, the number of points processed. The seasonal index of the next observation is `observed % period`.

- `initialize(y)` derives the starting state from the first two seasons (theory.md).
- `fit(y)` calls `initialize` and then `observe` for every point, so the state describes the end of the training window; there is no separate coefficient estimation.
- `next()` returns level + trend + seasonal[observed % period].
- `observe(y)` applies the three recursions and increments `observed`. Predict first, then observe, exactly as in modules 01-03.
- `save`/`load` use the shared archive with tag `HoltWinters_V1`; the archive contains the complete state, so a reloaded model forecasts the same next month.

`03_implementation.cpp` fits on all 144 AirPassengers months and replays the recursions from the initial state to write components.csv and level.svg, trend.svg, season.svg into its results folder.

`04_end_to_end.cpp` calls the shared workflow with `Forecaster::HoltWinters`. The workflow encodes the 27 candidates as an index 0..26 and decodes (alpha, beta, gamma) from base-3 digits; validation/candidate_scores.csv lists the three weights, the expanding origin and the fold RMSE. The selected model is saved to model/holt_winters_model.txt, reloaded and verified, then evaluated one step ahead on the held-out 20 percent with seasonal-naive and last-value references in the same metrics.json.
