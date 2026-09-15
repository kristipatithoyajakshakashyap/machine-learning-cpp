# Data understanding

Rows: 103. Features: 6. Target: passengers_thousands.

- lag_1: 0 missing, 0 IQR outliers. Mean 249.524, median 235. Investigate domain validity before removing outliers.
- lag_2: 0 missing, 0 IQR outliers. Mean 246.311, median 235. Investigate domain validity before removing outliers.
- lag_12: 0 missing, 1 IQR outliers. Mean 224.058, median 209. Investigate domain validity before removing outliers.
- month_index: 0 missing, 0 IQR outliers. Mean 63, median 63. Investigate domain validity before removing outliers.
- season_sin: 0 missing, 0 IQR outliers. Mean 0.0362335, median 3.67394e-16. Investigate domain validity before removing outliers.
- season_cos: 0 missing, 0 IQR outliers. Mean 3.44924e-17, median 4.8924e-16. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
