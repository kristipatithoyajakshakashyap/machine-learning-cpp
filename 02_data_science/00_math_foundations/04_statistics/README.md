# Sampling and uncertainty

## Purpose
Separates the spread of the data (standard deviation) from the uncertainty of an estimate (standard error) and demonstrates the bootstrap as a resampling approach to the same question. This is the vocabulary used by every later hypothesis test.

## Prerequisites
Module 03_probability.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_04` | sample mean and variance, standard error, bootstrap resampling | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/04_statistics/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_04
.\build\02_data_science\00_math_foundations\04_statistics\s00_math_04.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_04`: a nonzero exit code means a numerical acceptance check failed.

## dsts functions used
`dsts::Series`, `dsts::bootstrap_mean` (from `include/dsts/stats.hpp`).

## Theory
Sample variability is not uncertainty in the mean. Standard deviation describes observations; standard error approximately scales as s/sqrt(n). Bootstrap resampling approximates estimator variability under an IID assumption.

## Math intuition
For (2,4,6), mean=4, sample variance=((2-4)^2+0+(6-4)^2)/(3-1)=4. Standard error=2/sqrt(3).

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code; a nonzero exit means the
mathematical invariant failed.

## Exercise
Compare bootstrap spread for n=3 and n=30; explain why time series violate IID resampling.

## Key takeaways
- Standard deviation describes the data; standard error describes the estimate.
- The bootstrap only works when resampled rows are exchangeable (IID).

## Next module
Continue with **05_derivatives_and_gradients**.
