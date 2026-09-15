# Hierarchical clustering

Agglomerative clustering starts with individual observations and repeatedly merges the closest active clusters. Single linkage uses the nearest pair and can chain; complete linkage uses the farthest pair; average uses all pair distances. Ward merges minimize the increase in within-cluster squared error. The complete hierarchy supports multiple cuts without refitting. It has no native new-observation prediction rule.

Before modeling, inspect feature units, missingness, duplicates, outliers, distributions and correlations. The end-to-end lesson records this analysis and explains the metric limitations.
