# Data understanding

Rows: 344. Features: 4. Target: species.

- bill_length_mm: 2 missing, 0 IQR outliers. Mean 43.9219, median 44.45. Investigate domain validity before removing outliers.
- bill_depth_mm: 2 missing, 0 IQR outliers. Mean 17.1512, median 17.3. Investigate domain validity before removing outliers.
- flipper_length_mm: 2 missing, 0 IQR outliers. Mean 200.915, median 197. Investigate domain validity before removing outliers.
- body_mass_g: 2 missing, 0 IQR outliers. Mean 4201.75, median 4050. Investigate domain validity before removing outliers.

Duplicate feature rows: 1. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
