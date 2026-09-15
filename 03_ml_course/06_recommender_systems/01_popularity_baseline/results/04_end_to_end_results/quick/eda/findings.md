# Data understanding

Rows: 9601. Features: 3. Target: rating.

- timestamp: 0 missing, 0 IQR outliers. Mean 8.75375e+08, median 8.75313e+08. Investigate domain validity before removing outliers.
- user_training_ratings: 0 missing, 391 IQR outliers. Mean 157.384, median 153. Investigate domain validity before removing outliers.
- item_training_ratings: 0 missing, 188 IQR outliers. Mean 18.5063, median 15. Investigate domain validity before removing outliers.

Duplicate feature rows: 255. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
