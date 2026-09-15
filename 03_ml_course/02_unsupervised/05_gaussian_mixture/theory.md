# Gaussian mixtures

A Gaussian mixture models a density as a weighted sum of Gaussian components. Responsibilities are soft memberships; component labels are not necessarily semantic classes. EM alternates conditional responsibilities and weighted parameter updates. This implementation uses diagonal covariance, which assumes within-component conditional independence along the supplied axes. Initialization matters and likelihood can increase by collapsing a component without regularization.

Before modeling, inspect feature units, missingness, duplicates, outliers, distributions and correlations. The end-to-end lesson records this analysis and explains the metric limitations.
