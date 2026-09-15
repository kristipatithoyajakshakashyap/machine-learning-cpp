# Data understanding

Rows: 309. Features: 30. Target: diagnosis.

- radius_mean: 0 missing, 14 IQR outliers. Mean 12.5152, median 12.34. Investigate domain validity before removing outliers.
- texture_mean: 0 missing, 11 IQR outliers. Mean 18.4035, median 17.84. Investigate domain validity before removing outliers.
- perimeter_mean: 0 missing, 14 IQR outliers. Mean 80.6806, median 78.94. Investigate domain validity before removing outliers.
- area_mean: 0 missing, 13 IQR outliers. Mean 496.378, median 466.1. Investigate domain validity before removing outliers.
- smoothness_mean: 0 missing, 3 IQR outliers. Mean 0.0938557, median 0.09246. Investigate domain validity before removing outliers.
- compactness_mean: 0 missing, 10 IQR outliers. Mean 0.0859708, median 0.07698. Investigate domain validity before removing outliers.
- concavity_mean: 0 missing, 22 IQR outliers. Mean 0.055804, median 0.04006. Investigate domain validity before removing outliers.
- concave_points_mean: 0 missing, 20 IQR outliers. Mean 0.0309257, median 0.02541. Investigate domain validity before removing outliers.
- symmetry_mean: 0 missing, 7 IQR outliers. Mean 0.176372, median 0.1735. Investigate domain validity before removing outliers.
- fractal_dimension_mean: 0 missing, 12 IQR outliers. Mean 0.0628191, median 0.0613. Investigate domain validity before removing outliers.
- radius_se: 0 missing, 18 IQR outliers. Mean 0.308298, median 0.271. Investigate domain validity before removing outliers.
- texture_se: 0 missing, 11 IQR outliers. Mean 1.25327, median 1.14. Investigate domain validity before removing outliers.
- perimeter_se: 0 missing, 14 IQR outliers. Mean 2.17782, median 1.959. Investigate domain validity before removing outliers.
- area_se: 0 missing, 24 IQR outliers. Mean 24.5162, median 20.21. Investigate domain validity before removing outliers.
- smoothness_se: 0 missing, 15 IQR outliers. Mean 0.0073215, median 0.006547. Investigate domain validity before removing outliers.
- compactness_se: 0 missing, 19 IQR outliers. Mean 0.0232934, median 0.01764. Investigate domain validity before removing outliers.
- concavity_se: 0 missing, 22 IQR outliers. Mean 0.0286651, median 0.01954. Investigate domain validity before removing outliers.
- concave_points_se: 0 missing, 17 IQR outliers. Mean 0.0107113, median 0.009393. Investigate domain validity before removing outliers.
- symmetry_se: 0 missing, 12 IQR outliers. Mean 0.0207875, median 0.01924. Investigate domain validity before removing outliers.
- fractal_dimension_se: 0 missing, 17 IQR outliers. Mean 0.00374656, median 0.002961. Investigate domain validity before removing outliers.
- radius_worst: 0 missing, 15 IQR outliers. Mean 13.9097, median 13.57. Investigate domain validity before removing outliers.
- texture_worst: 0 missing, 3 IQR outliers. Mean 24.3118, median 23.31. Investigate domain validity before removing outliers.
- perimeter_worst: 0 missing, 14 IQR outliers. Mean 90.877, median 88.12. Investigate domain validity before removing outliers.
- area_worst: 0 missing, 15 IQR outliers. Mean 613.083, median 559.5. Investigate domain validity before removing outliers.
- smoothness_worst: 0 missing, 5 IQR outliers. Mean 0.127657, median 0.1275. Investigate domain validity before removing outliers.
- compactness_worst: 0 missing, 11 IQR outliers. Mean 0.201705, median 0.1788. Investigate domain validity before removing outliers.
- concavity_worst: 0 missing, 18 IQR outliers. Mean 0.192913, median 0.1514. Investigate domain validity before removing outliers.
- concave_points_worst: 0 missing, 11 IQR outliers. Mean 0.084542, median 0.07926. Investigate domain validity before removing outliers.
- symmetry_worst: 0 missing, 6 IQR outliers. Mean 0.27517, median 0.2722. Investigate domain validity before removing outliers.
- fractal_dimension_worst: 0 missing, 14 IQR outliers. Mean 0.0804327, median 0.07757. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
