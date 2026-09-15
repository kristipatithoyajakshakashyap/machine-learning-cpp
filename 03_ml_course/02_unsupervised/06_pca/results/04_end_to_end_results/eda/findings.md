# Data understanding

Rows: 275. Features: 4. Target: species.

- bill_length_mm: 2 missing, 0 IQR outliers. Mean 43.8385, median 44.1. Investigate domain validity before removing outliers.
- bill_depth_mm: 2 missing, 0 IQR outliers. Mean 17.1828, median 17.3. Investigate domain validity before removing outliers.
- flipper_length_mm: 2 missing, 0 IQR outliers. Mean 200.63, median 197. Investigate domain validity before removing outliers.
- body_mass_g: 2 missing, 0 IQR outliers. Mean 4193.5, median 4050. Investigate domain validity before removing outliers.

Duplicate feature rows: 1. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
