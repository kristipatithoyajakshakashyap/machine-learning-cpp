# Data understanding

Rows: 320. Features: 8. Target: median_house_value.

- longitude: 0 missing, 0 IQR outliers. Mean -119.627, median -118.59. Investigate domain validity before removing outliers.
- latitude: 0 missing, 0 IQR outliers. Mean 35.7089, median 34.28. Investigate domain validity before removing outliers.
- housing_median_age: 0 missing, 0 IQR outliers. Mean 28.6187, median 29. Investigate domain validity before removing outliers.
- total_rooms: 0 missing, 20 IQR outliers. Mean 2738.17, median 2270. Investigate domain validity before removing outliers.
- total_bedrooms: 5 missing, 17 IQR outliers. Mean 571.914, median 444. Investigate domain validity before removing outliers.
- population: 0 missing, 27 IQR outliers. Mean 1462.28, median 1199. Investigate domain validity before removing outliers.
- households: 0 missing, 14 IQR outliers. Mean 525.947, median 417. Investigate domain validity before removing outliers.
- median_income: 0 missing, 6 IQR outliers. Mean 3.80712, median 3.487. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
