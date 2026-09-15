# Implementation

`Arima` (Model.hpp) stores the orders p, d, q, the constant, the AR coefficients `phi`, the MA coefficients `theta`, the full observed `history` and the `residuals` of the differenced series.

- `difference(y, times)` is a static helper returning the times-th difference; it is also used by the tests and the lessons.
- `fit(y)` differences d times, solves the (p+1)-dimensional normal equations for c and phi by Gaussian elimination with partial pivoting, then runs `css_iterations` = 500 gradient steps on the conditional sum of squares for theta. The gradient uses the exact recursive derivative of each residual, the step is `css_learning_rate` / residual variance so it is scale free, and every theta is clipped to (-0.99, 0.99).
- `next()` computes the forecast of the next difference (constant + AR terms on the last p differences + MA terms on the last q residuals; missing values count as zero) and adds `integration_offset()`, the sum of the last value of each partially differenced series.
- `observe(y)` records the residual of the difference forecast and appends y to the history. Predict first, then observe.
- `save`/`load` use the shared archive with tag `Arima_V1`; the archive holds the history and residuals, so a reloaded model forecasts the same next month.

`03_implementation.cpp` fits ARIMA(1,1,1) on all 144 months and writes differences.csv, autocorrelation.csv (raw and differenced ACF up to lag 24) and the figures differences.svg, autocorrelation.svg, residuals.svg into its results folder.

`04_end_to_end.cpp` calls the shared workflow with `Forecaster::Arima`. The four candidates are encoded as an index 0..3 decoded to p = 1 + i / 2, q = 1 + i mod 2, d = 1; validation/candidate_scores.csv lists p, d, q with the fold RMSE. The selected model is saved to model/arima_model.txt, reloaded and verified, then evaluated one step ahead on the held-out 20 percent next to the seasonal-naive and last-value references.
