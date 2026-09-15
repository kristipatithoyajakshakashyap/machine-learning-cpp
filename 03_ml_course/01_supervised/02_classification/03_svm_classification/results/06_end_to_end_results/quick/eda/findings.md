# Data understanding

Rows: 319. Features: 30. Target: diagnosis.

- radius_mean: 0 missing, 5 IQR outliers. Mean 14.1013, median 13.34. Investigate domain validity before removing outliers.
- texture_mean: 0 missing, 3 IQR outliers. Mean 19.3021, median 18.91. Investigate domain validity before removing outliers.
- perimeter_mean: 0 missing, 5 IQR outliers. Mean 91.8111, median 86.49. Investigate domain validity before removing outliers.
- area_mean: 0 missing, 11 IQR outliers. Mean 651.539, median 546.4. Investigate domain validity before removing outliers.
- smoothness_mean: 0 missing, 4 IQR outliers. Mean 0.0964433, median 0.0961. Investigate domain validity before removing outliers.
- compactness_mean: 0 missing, 10 IQR outliers. Mean 0.10471, median 0.08995. Investigate domain validity before removing outliers.
- concavity_mean: 0 missing, 10 IQR outliers. Mean 0.0892939, median 0.05892. Investigate domain validity before removing outliers.
- concave_points_mean: 0 missing, 9 IQR outliers. Mean 0.0486773, median 0.03334. Investigate domain validity before removing outliers.
- symmetry_mean: 0 missing, 9 IQR outliers. Mean 0.182034, median 0.1799. Investigate domain validity before removing outliers.
- fractal_dimension_mean: 0 missing, 15 IQR outliers. Mean 0.0628791, median 0.06194. Investigate domain validity before removing outliers.
- radius_se: 0 missing, 20 IQR outliers. Mean 0.404201, median 0.3276. Investigate domain validity before removing outliers.
- texture_se: 0 missing, 10 IQR outliers. Mean 1.21116, median 1.142. Investigate domain validity before removing outliers.
- perimeter_se: 0 missing, 21 IQR outliers. Mean 2.85455, median 2.281. Investigate domain validity before removing outliers.
- area_se: 0 missing, 36 IQR outliers. Mean 39.9993, median 24.6. Investigate domain validity before removing outliers.
- smoothness_se: 0 missing, 15 IQR outliers. Mean 0.00691395, median 0.006174. Investigate domain validity before removing outliers.
- compactness_se: 0 missing, 21 IQR outliers. Mean 0.0255038, median 0.02083. Investigate domain validity before removing outliers.
- concavity_se: 0 missing, 13 IQR outliers. Mean 0.0323152, median 0.02575. Investigate domain validity before removing outliers.
- concave_points_se: 0 missing, 16 IQR outliers. Mean 0.011662, median 0.01076. Investigate domain validity before removing outliers.
- symmetry_se: 0 missing, 17 IQR outliers. Mean 0.0206041, median 0.01853. Investigate domain validity before removing outliers.
- fractal_dimension_se: 0 missing, 17 IQR outliers. Mean 0.00373446, median 0.00313. Investigate domain validity before removing outliers.
- radius_worst: 0 missing, 8 IQR outliers. Mean 16.32, median 14.97. Investigate domain validity before removing outliers.
- texture_worst: 0 missing, 3 IQR outliers. Mean 25.7743, median 25.58. Investigate domain validity before removing outliers.
- perimeter_worst: 0 missing, 7 IQR outliers. Mean 107.553, median 97.59. Investigate domain validity before removing outliers.
- area_worst: 0 missing, 18 IQR outliers. Mean 886.387, median 684.6. Investigate domain validity before removing outliers.
- smoothness_worst: 0 missing, 3 IQR outliers. Mean 0.132321, median 0.1323. Investigate domain validity before removing outliers.
- compactness_worst: 0 missing, 10 IQR outliers. Mean 0.256314, median 0.2057. Investigate domain validity before removing outliers.
- concavity_worst: 0 missing, 6 IQR outliers. Mean 0.276619, median 0.2241. Investigate domain validity before removing outliers.
- concave_points_worst: 0 missing, 0 IQR outliers. Mean 0.114542, median 0.09993. Investigate domain validity before removing outliers.
- symmetry_worst: 0 missing, 15 IQR outliers. Mean 0.291902, median 0.2819. Investigate domain validity before removing outliers.
- fractal_dimension_worst: 0 missing, 12 IQR outliers. Mean 0.0845, median 0.08075. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
