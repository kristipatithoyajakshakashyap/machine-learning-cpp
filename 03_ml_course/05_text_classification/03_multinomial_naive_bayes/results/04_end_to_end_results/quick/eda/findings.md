# Data understanding

Rows: 799. Features: 4. Target: spam.

- characters: 0 missing, 10 IQR outliers. Mean 75.3166, median 56. Investigate domain validity before removing outliers.
- tokens: 0 missing, 8 IQR outliers. Mean 15.2491, median 12. Investigate domain validity before removing outliers.
- digits: 0 missing, 93 IQR outliers. Mean 1.93617, median 0. Investigate domain validity before removing outliers.
- exclamations: 0 missing, 110 IQR outliers. Mean 0.192741, median 0. Investigate domain validity before removing outliers.

Duplicate feature rows: 226. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
