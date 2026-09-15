# Exercises

1. Change the initialisation to use only the first season (trend 0) and compare the rolling-origin scores; explain why the ranking of gamma values changes.
2. Implement the multiplicative recursions and rerun on AirPassengers; compare held-out RMSE with the additive fit on log(y).
3. Derive the h-step forecast l_t + h b_t + s_{t+h-m(k+1)} and add a `forecast(h)` method; evaluate a 12-step forecast from a single origin and contrast it with the one-step metrics.
4. Add a damped trend (multiply b by phi < 1 in the level and forecast equations) and grid phi with the other weights.
5. Plot the seasonal offsets stored at the end of training against the average monthly deviation in the data; explain the differences.
