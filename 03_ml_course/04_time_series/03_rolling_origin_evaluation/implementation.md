# Implementation

The baseline stores a seasonal period and observed history. LagRegression uses canonical ElasticNet with L1 ratio zero and the shared fitted Preprocessor. The rolling workflow selects regularization on expanding training windows, freezes the final fit, predicts each holdout month, and then appends its actual observation. CSV predictions retain month indices and split membership. Persisted models precede the holdout.
