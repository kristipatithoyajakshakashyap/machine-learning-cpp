# Theory

Convert an ordered series to supervised rows using lagged targets, time and seasonal features. A row for month t may use t-1, t-2 and t-12, never the target at t or later. Fit imputation/scaling and ridge regression on historical rows only.
