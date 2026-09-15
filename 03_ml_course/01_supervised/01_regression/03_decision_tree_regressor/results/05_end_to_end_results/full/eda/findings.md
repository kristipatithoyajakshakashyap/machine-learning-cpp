# Data understanding

Rows: 16512. Features: 8. Target: median_house_value.

- longitude: 0 missing, 0 IQR outliers. Mean -119.577, median -118.51. Investigate domain validity before removing outliers.
- latitude: 0 missing, 0 IQR outliers. Mean 35.6353, median 34.26. Investigate domain validity before removing outliers.
- housing_median_age: 0 missing, 0 IQR outliers. Mean 28.5704, median 29. Investigate domain validity before removing outliers.
- total_rooms: 0 missing, 1026 IQR outliers. Mean 2639.91, median 2127. Investigate domain validity before removing outliers.
- total_bedrooms: 165 missing, 1026 IQR outliers. Mean 538.535, median 436. Investigate domain validity before removing outliers.
- population: 0 missing, 958 IQR outliers. Mean 1429.86, median 1170. Investigate domain validity before removing outliers.
- households: 0 missing, 986 IQR outliers. Mean 500.196, median 410. Investigate domain validity before removing outliers.
- median_income: 0 missing, 562 IQR outliers. Mean 3.87625, median 3.5313. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
