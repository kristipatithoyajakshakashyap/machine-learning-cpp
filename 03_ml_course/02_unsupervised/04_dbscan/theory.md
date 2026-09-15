# DBSCAN

DBSCAN forms clusters through density reachability. A core point has at least min_samples observations, including itself, within eps. A border point is near a core point but is not core. Remaining observations are noise. Connected core neighborhoods can trace non-convex shapes. One global radius struggles with varying density; distance concentration makes high-dimensional density less useful.

Before modeling, inspect feature units, missingness, duplicates, outliers, distributions and correlations. The end-to-end lesson records this analysis and explains the metric limitations.
