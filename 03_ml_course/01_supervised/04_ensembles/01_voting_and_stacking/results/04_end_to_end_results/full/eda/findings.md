# Data understanding

Rows: 141. Features: 13. Target: class.

- alcohol: 0 missing, 0 IQR outliers. Mean 12.9771, median 13.03. Investigate domain validity before removing outliers.
- malic_acid: 0 missing, 1 IQR outliers. Mean 2.31312, median 1.83. Investigate domain validity before removing outliers.
- ash: 0 missing, 4 IQR outliers. Mean 2.37184, median 2.36. Investigate domain validity before removing outliers.
- alcalinity: 0 missing, 3 IQR outliers. Mean 19.539, median 19.5. Investigate domain validity before removing outliers.
- magnesium: 0 missing, 4 IQR outliers. Mean 99.9078, median 98. Investigate domain validity before removing outliers.
- total_phenols: 0 missing, 0 IQR outliers. Mean 2.29716, median 2.3. Investigate domain validity before removing outliers.
- flavanoids: 0 missing, 0 IQR outliers. Mean 2.02546, median 2.11. Investigate domain validity before removing outliers.
- nonflavanoid_phenols: 0 missing, 0 IQR outliers. Mean 0.360355, median 0.34. Investigate domain validity before removing outliers.
- proanthocyanins: 0 missing, 3 IQR outliers. Mean 1.57702, median 1.54. Investigate domain validity before removing outliers.
- color_intensity: 0 missing, 2 IQR outliers. Mean 4.97582, median 4.68. Investigate domain validity before removing outliers.
- hue: 0 missing, 1 IQR outliers. Mean 0.95522, median 0.96. Investigate domain validity before removing outliers.
- od280: 0 missing, 0 IQR outliers. Mean 2.59652, median 2.78. Investigate domain validity before removing outliers.
- proline: 0 missing, 0 IQR outliers. Mean 746.184, median 660. Investigate domain validity before removing outliers.

Duplicate feature rows: 0. Duplicate features do not necessarily mean duplicate observations.

Correlation is descriptive, not causal. Supervised workflows produce this EDA on training rows only; preprocessing is independently fitted inside each CV fold.
