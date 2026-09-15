# Implementation

The implementation estimates class means, empirical priors and pooled covariance. Cholesky triangular solves compute Mahalanobis distances without explicitly inverting covariance. Log-sum-exp normalization produces stable probabilities. Labels need not be consecutive integers.
