# Implementation

QDA reuses the canonical Gaussian discriminant implementation with class-specific maximum-likelihood covariances. It retains separate Cholesky factors and log determinants for every class. Save/load includes label mapping and all fitted state.
