# Implementation walkthrough

PCA.hpp uses Jacobi plane rotations to diagonalize the symmetric covariance matrix, sorts eigenpairs, then stores the training mean and selected axes. transform always subtracts the fitted mean; inverse_transform adds it back. The project selects 95% training variance and evaluates reconstruction only on held-out observations.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.

Covariance construction costs O(np^2) and stores O(p^2) values. The transparent maximum-entry Jacobi routine is intended for small dense feature sets; it is not a sparse or large-scale eigensolver. Eigenvector signs are arbitrary, so compare reconstructions or projection subspaces when checking another implementation.
