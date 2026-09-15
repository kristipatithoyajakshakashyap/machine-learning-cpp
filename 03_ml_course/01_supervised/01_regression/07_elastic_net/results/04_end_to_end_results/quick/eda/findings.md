# Data understanding

Rows: 320. Features: 10. Target: target.

- age: 0 missing, 0 IQR outliers. Mean 48.65, median 50. Investigate domain validity before removing outliers.
- sex: 0 missing, 0 IQR outliers. Mean 1.4875, median 1. Investigate domain validity before removing outliers.
- bmi: 0 missing, 6 IQR outliers. Mean 26.1241, median 25.55. Investigate domain validity before removing outliers.
- bp: 0 missing, 1 IQR outliers. Mean 94.5708, median 93. Investigate domain validity before removing outliers.
- s1: 0 missing, 2 IQR outliers. Mean 190.1, median 187. Investigate domain validity before removing outliers.
- s2: 0 missing, 2 IQR outliers. Mean 116.442, median 113.7. Investigate domain validity before removing outliers.
- s3: 0 missing, 7 IQR outliers. Mean 50.1656, median 49. Investigate domain validity before removing outliers.
- s4: 0 missing, 1 IQR outliers. Mean 4.04856, median 4. Investigate domain validity before removing outliers.
- s5: 0 missing, 5 IQR outliers. Mean 4.62631, median 4.585. Investigate domain validity before removing outliers.
- s6: 0 missing, 12 IQR outliers. Mean 90.8281, median 91. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
