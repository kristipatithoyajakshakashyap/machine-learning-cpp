# Eigen-decomposition and SVD

## Purpose
Exercises the two matrix factorisations that power PCA: the Jacobi symmetric eigen-solver and the thin SVD built on top of it. The worked example checks eigenpair residuals, the trace identity, orthonormality and rank-1 reconstruction (Eckart-Young).

## Prerequisites
Modules 02_linear_algebra and 06_optimization.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_07` | symmetric eigen-decomposition, thin SVD, orthonormality, low-rank approximation | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/07_eigen_svd/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_07
.\build\02_data_science\00_math_foundations\07_eigen_svd\s00_math_07.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_07`. A nonzero exit code means a numerical acceptance check failed.

## dsts functions used
`dsts::symmetric_eigen`, `dsts::svd_thin` (from `include/dsts/pca.hpp`). PCA reuses the same solver on the covariance matrix.

## Theory
A symmetric matrix A has real eigenvalues and orthogonal eigenvectors: A v = lambda v. Cyclic Jacobi rotations zero one off-diagonal pair at a time until A becomes diagonal. The accumulated rotations are the eigenvectors. Any matrix A (m x n) factors as A = U diag(S) V^T. The right singular vectors V are eigenvectors of A^T A and S = sqrt(eigenvalues).

## Math intuition
For A = [[4,1,0],[1,3,1],[0,1,2]] every eigenpair must satisfy ||A v - lambda v|| ~ 0 and the eigenvalues must sum to trace(A) = 9. For the 4x2 matrix in the lesson, U diag(S) V^T rebuilds A to machine precision, and dropping the smallest singular value gives the best rank-1 approximation (Eckart-Young).

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code. A nonzero exit means the
mathematical invariant failed.

## Exercise
Replace the 4x2 matrix by a rank-1 matrix and confirm the second singular value is ~0. Compare rank-1 reconstruction error with the dropped singular value.

## Key takeaways
- Eigenvalues of a symmetric matrix sum to its trace. Eigenvectors are orthonormal.
- Truncating the SVD gives the best low-rank approximation in the Frobenius norm.

## Next module
You have finished the mathematical bridge. Continue with 01_series_dataframe.