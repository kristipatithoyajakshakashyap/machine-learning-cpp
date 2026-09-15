# Theory

## Stationarity

ARMA models assume a weakly stationary series: constant mean, constant variance and autocovariances that depend only on the lag. AirPassengers violates all three (rising level, growing seasonal amplitude, strong lag-12 correlation). Differencing removes the trend part: w_t = y_t - y_{t-1} has roughly constant mean. Differencing twice removes a quadratic trend; differencing at lag 12 would remove the seasonal pattern (seasonal ARIMA, not implemented here). Over-differencing introduces a strong negative lag-1 autocorrelation and inflates variance, so d should be as small as possible.

## The model

ARIMA(p, d, q) says that the d-th difference w_t follows

w_t = c + phi_1 w_{t-1} + ... + phi_p w_{t-p} + e_t + theta_1 e_{t-1} + ... + theta_q e_{t-q}

with e_t white noise. The AR part carries persistence: a large phi_1 means that a high difference tends to be followed by another high one. The MA part corrects for recent surprises: a positive theta_1 means that yesterday's unexpected jump is partly repeated today. The constant c is the drift of the differenced series (about 2.3 passengers per month here).

## Reading ACF and PACF

- The autocorrelation function (ACF) of a pure MA(q) process cuts off after lag q; of a pure AR(p) process it decays geometrically.
- The partial autocorrelation function (PACF, the coefficient of lag k in an AR(k) regression) of a pure AR(p) process cuts off after lag p; of an MA(q) process it decays.
- For a non-stationary series the ACF decays very slowly; this is the visual cue to difference. `forecast_arima_implementation` writes both raw and differenced ACFs; the differenced one still spikes at lag 12 because ordinary differencing does not remove seasonality.

## Estimation in this module

1. Difference d times.
2. Fit the AR coefficients and the constant by ordinary least squares of w_t on its p lags (Yule-Walker would be the alternative).
3. With the AR part fixed, minimise the conditional sum of squares S(theta) = sum e_t^2 where e_t is computed recursively from the data (e_t = 0 before the sample). Plain gradient descent with an exact recursive derivative, 500 iterations, a step scaled by the residual variance, and |theta| clipped below 0.99 to stay invertible.
4. Keep the residual history so `next()` can use e_{t-1}, ..., e_{t-q}, and `observe()` appends the new residual.

## Limits of this estimator

- Fitting AR first and MA second is not joint maximum likelihood; the AR coefficients absorb some of what the MA term should explain, and the estimates are biased for mixed models on short series.
- The conditional (not exact) sum of squares ignores the distribution of the first observations, which matters with 115 training months.
- Gradient descent with a fixed budget may stop short of the optimum for strongly correlated errors; there is no standard error, no AIC and no automatic order selection beyond the rolling-origin grid.
- No seasonal terms: the lag-12 structure is left in the residuals. This is why a well-chosen seasonal-naive or Holt-Winters model can beat the ARIMA orders used here.

## When to prefer ARIMA

ARIMA is a good choice for non-seasonal series with short-memory autocorrelation, for series that become stationary after one or two differences, and when a probabilistic interpretation of the innovations is wanted. For strongly seasonal data reach for seasonal ARIMA, Holt-Winters or lag regression with seasonal features.
