# Math intuition

## Differencing by hand

Series 5, 8, 12, 15, 19, 22, 26, 29. First differences w = 3, 4, 3, 4, 3, 4, 3. The trend of about 3.5 per step is gone; what remains is an alternating pattern, so the differenced series has negative lag-1 autocorrelation. A second difference would be 1, -1, 1, -1, 1, -1: no trend either, but larger swings. This is the over-differencing warning in theory.md.

## AR(1) by least squares

Regress w_t on w_{t-1} with a constant: rows (w_{t-1}, w_t) = (3,4), (4,3), (3,4), (4,3), (3,4), (4,3). The normal equations give phi = -1 and c = 7, because 4 = 7 - 3 and 3 = 7 - 4 exactly. `forecast_arima_math_intuition` prints these numbers and the integrated forecast: next difference c + phi * 3 = 4, next level 29 + 4 = 33.

## Integration

If w = delta y, then y_{t+1} = y_t + w_{t+1}. For d = 2, delta y_{t+1} = delta y_t + w_{t+1} and y_{t+1} = y_t + delta y_{t+1}. In general the forecast of the d-th difference plus the last value of every lower difference gives the forecast of y; the model computes this sum in `integration_offset()`. The test checks it on squares: second differences are all 2, so ARIMA(0,2,0) predicts 121 after 100.

## MA(1) and the residual recursion

With theta known, residuals are e_t = w_t - c - phi w_{t-1} - theta e_{t-1}, starting from e_0 = 0. The residual at time t depends on every earlier residual, so d e_t / d theta = -e_{t-1} - theta d e_{t-1} / d theta. This recursion is what `css_residuals` accumulates, and it is why the objective is a conditional (on e_0 = 0) sum of squares.

## ACF and PACF as fingerprints

For AR(1) with phi = 0.7 the ACF is 0.7, 0.49, 0.343, ... (geometric decay) and the PACF is 0.7 at lag 1 and 0 afterwards. For MA(1) with theta = 0.5 the ACF is theta / (1 + theta^2) = 0.4 at lag 1 and 0 afterwards, while the PACF decays. Comparing the shapes of the two functions is the classical way to guess p and q before estimating; the workflow replaces the guess with rolling-origin validation over a small grid.

## What the AirPassengers fit looks like

After one difference the drift c is a couple of passengers per month, phi_1 is small and positive, and the lag-12 autocorrelation of the differences stays near 0.7. ARIMA(1,1,1) therefore forecasts the last value plus drift plus a small correction; it cannot anticipate the summer peak. The seasonal-naive reference in metrics.json makes that limitation visible.
