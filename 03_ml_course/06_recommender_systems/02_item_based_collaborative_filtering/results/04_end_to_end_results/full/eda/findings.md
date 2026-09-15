# Data understanding

Rows: 80003. Features: 3. Target: rating.

- timestamp: 0 missing, 0 IQR outliers. Mean 8.81566e+08, median 8.80845e+08. Investigate domain validity before removing outliers.
- user_training_ratings: 0 missing, 1946 IQR outliers. Mean 199.951, median 174. Investigate domain validity before removing outliers.
- item_training_ratings: 0 missing, 1709 IQR outliers. Mean 136.779, median 117. Investigate domain validity before removing outliers.

Duplicate feature rows: 471. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
