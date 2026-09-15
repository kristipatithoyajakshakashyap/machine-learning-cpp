# Data understanding

Rows: 240. Features: 10. Target: target.

- age: 0 missing, 0 IQR outliers. Mean 48.4542, median 50. Investigate domain validity before removing outliers.
- sex: 0 missing, 0 IQR outliers. Mean 1.50417, median 2. Investigate domain validity before removing outliers.
- bmi: 0 missing, 3 IQR outliers. Mean 26.2396, median 25.55. Investigate domain validity before removing outliers.
- bp: 0 missing, 0 IQR outliers. Mean 94.7277, median 93. Investigate domain validity before removing outliers.
- s1: 0 missing, 6 IQR outliers. Mean 187.383, median 184. Investigate domain validity before removing outliers.
- s2: 0 missing, 5 IQR outliers. Mean 114.214, median 111. Investigate domain validity before removing outliers.
- s3: 0 missing, 3 IQR outliers. Mean 49.8896, median 49. Investigate domain validity before removing outliers.
- s4: 0 missing, 1 IQR outliers. Mean 4.00471, median 4. Investigate domain validity before removing outliers.
- s5: 0 missing, 4 IQR outliers. Mean 4.60978, median 4.585. Investigate domain validity before removing outliers.
- s6: 0 missing, 6 IQR outliers. Mean 91.4375, median 92. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
