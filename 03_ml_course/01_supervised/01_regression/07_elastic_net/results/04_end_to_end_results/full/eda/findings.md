# Data understanding

Rows: 353. Features: 10. Target: target.

- age: 0 missing, 0 IQR outliers. Mean 48.1133, median 50. Investigate domain validity before removing outliers.
- sex: 0 missing, 0 IQR outliers. Mean 1.48725, median 1. Investigate domain validity before removing outliers.
- bmi: 0 missing, 2 IQR outliers. Mean 26.3385, median 25.7. Investigate domain validity before removing outliers.
- bp: 0 missing, 0 IQR outliers. Mean 95.0415, median 93. Investigate domain validity before removing outliers.
- s1: 0 missing, 7 IQR outliers. Mean 188.402, median 186. Investigate domain validity before removing outliers.
- s2: 0 missing, 5 IQR outliers. Mean 114.758, median 112.4. Investigate domain validity before removing outliers.
- s3: 0 missing, 5 IQR outliers. Mean 49.9193, median 48. Investigate domain validity before removing outliers.
- s4: 0 missing, 2 IQR outliers. Mean 4.03629, median 4. Investigate domain validity before removing outliers.
- s5: 0 missing, 5 IQR outliers. Mean 4.62753, median 4.585. Investigate domain validity before removing outliers.
- s6: 0 missing, 10 IQR outliers. Mean 91.2153, median 91. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
