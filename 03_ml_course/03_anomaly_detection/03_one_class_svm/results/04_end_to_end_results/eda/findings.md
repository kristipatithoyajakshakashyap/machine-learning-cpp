# Data understanding

Rows: 206. Features: 4. Target: species.

- bill_length_mm: 1 missing, 0 IQR outliers. Mean 43.9283, median 44.4. Investigate domain validity before removing outliers.
- bill_depth_mm: 1 missing, 0 IQR outliers. Mean 17.2185, median 17.5. Investigate domain validity before removing outliers.
- flipper_length_mm: 1 missing, 0 IQR outliers. Mean 200.722, median 197. Investigate domain validity before removing outliers.
- body_mass_g: 1 missing, 0 IQR outliers. Mean 4188.41, median 4050. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
