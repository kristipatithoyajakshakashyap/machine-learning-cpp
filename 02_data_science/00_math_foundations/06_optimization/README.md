# Optimization and regularization

## Purpose
Runs gradient descent on a one-dimensional quadratic with an L2 penalty and compares the iterate with the closed-form optimum. It shows concretely how a penalty shrinks the solution and how the step size controls convergence.

## Prerequisites
Module 05_derivatives_and_gradients.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_06` | gradient descent loop, learning rate, L2 regularisation, closed-form check | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/06_optimization/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_06
.\build\02_data_science\00_math_foundations\06_optimization\s00_math_06.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_06`. A nonzero exit code means a numerical acceptance check failed.

## dsts functions used
None. Standard library only.

## Theory
Gradient descent subtracts learning_rate times the gradient. Regularization changes the objective and its derivative. Convergence depends on step size and curvature.

## Math intuition
Minimize L=(w-3)^2+lambda*w^2. The gradient is 2(w-3)+2*lambda*w, and the exact optimum is 3/(1+lambda). At lambda=1 the optimum shrinks from 3 to 1.5.

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code. A nonzero exit means the
mathematical invariant failed.

## Exercise
Demonstrate divergence with a too-large step and compare early stopping with regularization.

## Key takeaways
- Gradient descent: w <- w - lr * dL/dw. The step size must respect the curvature.
- L2 regularisation shrinks the optimum toward zero by a factor 1/(1+lambda).

## Next module
Continue with 07_eigen_svd.