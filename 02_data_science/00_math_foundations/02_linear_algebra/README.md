# Linear algebra and projection

## Purpose
Shows how a vector splits into a component along a direction plus a perpendicular residual, which is the geometric idea behind least squares and PCA. The worked example projects points onto a unit axis and then runs a tiny PCA with `dsts::pca`.

## Prerequisites
Module 01_vectors_and_matrices.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_02` | projection onto a unit vector, residuals, Pythagorean identity, covariance eigenvector via PCA | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/02_linear_algebra/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_02
.\build\02_data_science\00_math_foundations\02_linear_algebra\s00_math_02.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_02`. A nonzero exit code means a numerical acceptance check failed.

## dsts functions used
`dsts::pca`, `dsts::pca_transform` (from `include/dsts/pca.hpp`).

## Theory
Orthogonal directions separate independent coordinates. A projection keeps the component parallel to an axis. Residuals are perpendicular. PCA extends this to covariance eigenvectors.

## Math intuition
Projection of x onto unit u is (x dot u)u. For x=(3,4), u=(1,0), projection=(3,0) and residual=(0,4). The Pythagorean identity gives 25=9+16.

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code. A nonzero exit means the
mathematical invariant failed.

## Exercise
Derive the covariance matrix by hand and verify its leading eigenvector up to sign.

## Key takeaways
- Projection = (x dot u) u for a unit vector u. The residual is orthogonal to u.
- PCA picks the axes that keep the most variance. Eigenvectors are only defined up to sign.

## Next module
Continue with 03_probability (conditional probability and Bayes rule).