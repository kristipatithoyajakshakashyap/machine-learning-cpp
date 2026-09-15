# Exercises

1. Compute the PACF of the differenced AirPassengers series up to lag 24 (fit AR(k) for k = 1..24 and keep the last coefficient) and compare its shape with the ACF written by `forecast_arima_implementation`.
2. Add a seasonal difference (lag 12) before the ordinary difference and rerun the grid; explain the change in held-out RMSE and in the residual autocorrelation at lag 12.
3. Replace the two-stage estimator with joint gradient descent over c, phi and theta and check whether the AR(1) recovery test still passes within 0.1.
4. Fit the models on log(y) and integrate the forecast back with exp; compare with module 04.
5. Estimate a one-step forecast interval from the residual variance and count how many held-out months fall inside a 95 percent band.
