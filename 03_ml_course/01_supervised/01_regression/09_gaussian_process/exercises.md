# Exercises

Sweep noise_var in `03_implementation.cpp` and record how the posterior band width at the training points changes. Compare the length scale chosen by cross-validation in `04_end_to_end.cpp` with the one that maximises the marginal-likelihood curve. Add a Matern-3/2 kernel and check whether the holdout coverage in `predictive_std.csv` moves closer to 95%. Measure fit time as the subsample grows from 100 to 400 rows and confirm the cubic trend.
