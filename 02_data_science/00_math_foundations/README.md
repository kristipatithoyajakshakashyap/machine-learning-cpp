# Module 00 — Mathematical bridge

## Purpose
Seven short, self-checking lessons that cover the mathematics the data-science
and ML tracks rely on: vectors and matrices, projection, probability, sampling
uncertainty, derivatives, gradient descent and matrix factorisation. Each lesson
computes a worked example, asserts the numerical result and saves it locally.

## Prerequisites
Track 01_get_to_know (C++ basics, `std::vector`, functions). No dsts knowledge
is required; a few lessons call `dsts::pca`, `dsts::bootstrap_mean`,
`dsts::symmetric_eigen` and `dsts::svd_thin`.

## Lessons

Complete the seven modules in numerical order. Every module has one source
file, `01_worked_example.cpp`, that writes `worked_example.csv` under
`results/01_worked_example_results/` inside that module.

| Module | Target | What it teaches | What it writes |
|--------|--------|-----------------|----------------|
| `01_vectors_and_matrices` | `s00_math_01` | vectors, matrices, dot and matrix-vector products | `worked_example.csv` |
| `02_linear_algebra` | `s00_math_02` | projection, residuals, PCA as covariance eigenvectors | `worked_example.csv` |
| `03_probability` | `s00_math_03` | conditional probability, Bayes rule, base rates | `worked_example.csv` |
| `04_statistics` | `s00_math_04` | mean, variance, standard error, bootstrap | `worked_example.csv` |
| `05_derivatives_and_gradients` | `s00_math_05` | analytic vs central-difference derivatives | `worked_example.csv` |
| `06_optimization` | `s00_math_06` | gradient descent and L2 regularisation | `worked_example.csv` |
| `07_eigen_svd` | `s00_math_07` | symmetric eigen-decomposition and thin SVD | `worked_example.csv` |

## Build and run

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_01
.\build\02_data_science\00_math_foundations\01_vectors_and_matrices\s00_math_01.exe
```

Each target is also a ctest (`ctest --test-dir build -R s00_math`); a nonzero
exit code means one of the asserted invariants failed.

## Key takeaways
- Every estimator, loss and model in the later tracks reduces to these operations.
- Numerical checks (residuals, trace identities, finite differences) are cheap
  and catch most implementation mistakes.

## Next module
Continue with **01_series_dataframe**.
