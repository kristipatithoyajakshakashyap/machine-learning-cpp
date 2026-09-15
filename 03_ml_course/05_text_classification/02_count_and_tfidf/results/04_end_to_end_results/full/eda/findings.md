# Data understanding

Rows: 4136. Features: 4. Target: spam.

- characters: 0 missing, 48 IQR outliers. Mean 79.646, median 61. Investigate domain validity before removing outliers.
- tokens: 0 missing, 60 IQR outliers. Mean 16.0317, median 12. Investigate domain validity before removing outliers.
- digits: 0 missing, 564 IQR outliers. Mean 2.19221, median 0. Investigate domain validity before removing outliers.
- exclamations: 0 missing, 675 IQR outliers. Mean 0.242021, median 0. Investigate domain validity before removing outliers.

Duplicate feature rows: 1977. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
