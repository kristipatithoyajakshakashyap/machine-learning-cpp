# Derivatives and gradients

## Purpose
Defines the derivative as local sensitivity and shows how a central finite difference independently checks an analytic derivative. Gradient checks of this kind are how the ML track validates back-propagation.

## Prerequisites
Module 01_vectors_and_matrices.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_05` | analytic derivative, central finite difference, truncation error | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/05_derivatives_and_gradients/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_05
.\build\02_data_science\00_math_foundations\05_derivatives_and_gradients\s00_math_05.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_05`. A nonzero exit code means a numerical acceptance check failed.

## dsts functions used
None. Standard library only.

## Theory
A derivative is local sensitivity. A gradient stacks partial derivatives and points toward greatest local increase. Central differences independently check an analytic derivative.

## Math intuition
For f(w)=(w-3)^2, df/dw=2(w-3). At w=1 the derivative is -4. The central difference [f(w+h)-f(w-h)]/(2h) approximates it with O(h^2) truncation error.

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code. A nonzero exit means the
mathematical invariant failed.

## Exercise
Sweep h from 1e-1 to 1e-16 and explain cancellation versus truncation error.

## Key takeaways
- A central difference has O(h^2) error. Too small an h reintroduces rounding error.
- Compare analytic and numerical gradients before trusting an optimiser.

## Next module
Continue with 06_optimization (gradient descent and regularisation).