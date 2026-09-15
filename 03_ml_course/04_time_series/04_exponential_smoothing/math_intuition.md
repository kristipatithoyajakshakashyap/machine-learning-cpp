# Math intuition

## A worked trace (season 4)

Series 10, 20, 30, 20, 12, 22, 32, 22. Season means: 20 and 22. Initial state: level 20, trend (22 - 20)/4 = 0.5, seasonal offsets (-10, 0, 10, 0) averaged from both seasons. With alpha = 0.5, beta = 0.1, gamma = 0.3:

- Step t = 0 forecasts 20 + 0.5 - 10 = 10.5. Observed 10. Level = 0.5 (10 + 10) + 0.5 (20.5) = 20.25; trend = 0.1 (0.25) + 0.9 (0.5) = 0.475; season[0] = 0.3 (10 - 20.25) + 0.7 (-10) = -10.075.
- Step t = 1 forecasts 20.25 + 0.475 + 0 = 20.725. Observed 20. Level = 0.5 (20) + 0.5 (20.725) = 20.3625, and so on.

`forecast_hw_math_intuition` prints the full table and checks that `fit()` reproduces the same end state as the manual trace.

## Reading the recursions

- The term (y_t - s_{t-m}) is the deseasonalised observation. The level is a moving average of deseasonalised values, corrected by the trend.
- The trend is a moving average of successive level differences.
- The seasonal update compares the observation with the new level, so the seasonal offset absorbs whatever the level and trend do not explain.
- All three are exponentially weighted averages: the weight on an observation k steps back is alpha (1 - alpha)^k for the level.

## Why the errors cancel

Write e_t = y_t - (l_{t-1} + b_{t-1} + s_{t-m}) for the one-step error. Then l_t = l_{t-1} + b_{t-1} + alpha e_t, b_t = b_{t-1} + alpha beta e_t and s_t = s_{t-m} + gamma (1 - alpha) e_t. Each state moves in proportion to the same forecast error, which is the error-correction view of exponential smoothing and the link to the innovations state-space form (ETS(A,A,A)).

## Choosing weights

Large alpha tracks level shifts quickly but passes noise into the forecast; large beta lets the slope react to short runs; large gamma rewrites the seasonal pattern each year. The workflow grid {0.2, 0.5, 0.8}^3 is coarse on purpose: with 115 training months the rolling-origin RMSE surface is flat enough that finer grids mostly fit validation noise.
