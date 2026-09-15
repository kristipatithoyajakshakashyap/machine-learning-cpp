# K-means

K-means partitions Euclidean observations around arithmetic centroids. Lloyd updates alternate nearest-center assignment and cluster means, reducing the within-cluster sum of squares. K-means++ samples later centers proportional to squared distance to the closest existing center. Multiple seeded restarts reduce local-minimum risk. Scaling changes the objective; elongated, unequal-density clusters can violate the spherical-cluster approximation.

Before modeling, inspect feature units, missingness, duplicates, outliers, distributions and correlations. The end-to-end lesson records this analysis and explains the metric limitations.
