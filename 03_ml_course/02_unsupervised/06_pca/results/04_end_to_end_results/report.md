# PCA project

Choose the smallest component count retaining 95% of training variance. Both imputation/scaling and projection are fitted only on training observations. Report reconstruction error on untouched rows in standardized units. High variance does not imply predictive relevance; compare downstream performance in a training-only validation pipeline before adopting PCA.
