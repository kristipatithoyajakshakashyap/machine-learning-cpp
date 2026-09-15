# Data understanding

Rows: 454. Features: 30. Target: diagnosis.

- radius_mean: 0 missing, 10 IQR outliers. Mean 14.1752, median 13.36. Investigate domain validity before removing outliers.
- texture_mean: 0 missing, 7 IQR outliers. Mean 19.4828, median 18.985. Investigate domain validity before removing outliers.
- perimeter_mean: 0 missing, 11 IQR outliers. Mean 92.3148, median 86.11. Investigate domain validity before removing outliers.
- area_mean: 0 missing, 19 IQR outliers. Mean 658.391, median 548.75. Investigate domain validity before removing outliers.
- smoothness_mean: 0 missing, 3 IQR outliers. Mean 0.0968345, median 0.096425. Investigate domain validity before removing outliers.
- compactness_mean: 0 missing, 11 IQR outliers. Mean 0.105718, median 0.095275. Investigate domain validity before removing outliers.
- concavity_mean: 0 missing, 13 IQR outliers. Mean 0.0909939, median 0.06265. Investigate domain validity before removing outliers.
- concave_points_mean: 0 missing, 9 IQR outliers. Mean 0.0499, median 0.03395. Investigate domain validity before removing outliers.
- symmetry_mean: 0 missing, 14 IQR outliers. Mean 0.181707, median 0.17995. Investigate domain validity before removing outliers.
- fractal_dimension_mean: 0 missing, 13 IQR outliers. Mean 0.0628226, median 0.061515. Investigate domain validity before removing outliers.
- radius_se: 0 missing, 30 IQR outliers. Mean 0.406326, median 0.3276. Investigate domain validity before removing outliers.
- texture_se: 0 missing, 18 IQR outliers. Mean 1.23927, median 1.15. Investigate domain validity before removing outliers.
- perimeter_se: 0 missing, 30 IQR outliers. Mean 2.86814, median 2.309. Investigate domain validity before removing outliers.
- area_se: 0 missing, 50 IQR outliers. Mean 39.9778, median 24.7. Investigate domain validity before removing outliers.
- smoothness_se: 0 missing, 24 IQR outliers. Mean 0.00713787, median 0.0064085. Investigate domain validity before removing outliers.
- compactness_se: 0 missing, 22 IQR outliers. Mean 0.0261033, median 0.021. Investigate domain validity before removing outliers.
- concavity_se: 0 missing, 18 IQR outliers. Mean 0.0327819, median 0.026335. Investigate domain validity before removing outliers.
- concave_points_se: 0 missing, 17 IQR outliers. Mean 0.0120087, median 0.011115. Investigate domain validity before removing outliers.
- symmetry_se: 0 missing, 25 IQR outliers. Mean 0.0206635, median 0.018795. Investigate domain validity before removing outliers.
- fractal_dimension_se: 0 missing, 23 IQR outliers. Mean 0.00384151, median 0.0032935. Investigate domain validity before removing outliers.
- radius_worst: 0 missing, 8 IQR outliers. Mean 16.3145, median 14.965. Investigate domain validity before removing outliers.
- texture_worst: 0 missing, 4 IQR outliers. Mean 25.9526, median 25.47. Investigate domain validity before removing outliers.
- perimeter_worst: 0 missing, 10 IQR outliers. Mean 107.595, median 97.665. Investigate domain validity before removing outliers.
- area_worst: 0 missing, 21 IQR outliers. Mean 881.593, median 683.95. Investigate domain validity before removing outliers.
- smoothness_worst: 0 missing, 6 IQR outliers. Mean 0.133256, median 0.13155. Investigate domain validity before removing outliers.
- compactness_worst: 0 missing, 12 IQR outliers. Mean 0.258371, median 0.21655. Investigate domain validity before removing outliers.
- concavity_worst: 0 missing, 10 IQR outliers. Mean 0.277509, median 0.2262. Investigate domain validity before removing outliers.
- concave_points_worst: 0 missing, 0 IQR outliers. Mean 0.116163, median 0.1011. Investigate domain validity before removing outliers.
- symmetry_worst: 0 missing, 19 IQR outliers. Mean 0.291091, median 0.28165. Investigate domain validity before removing outliers.
- fractal_dimension_worst: 0 missing, 21 IQR outliers. Mean 0.0841645, median 0.080245. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
