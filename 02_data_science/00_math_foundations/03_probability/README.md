# Probability and Bayes rule

## Purpose
Works through conditional probability and Bayes rule with a classic base-rate example so that later lessons on likelihoods and classifiers have a firm footing. The example is an arithmetic exercise, not a diagnostic tool.

## Prerequisites
Module 01_vectors_and_matrices (for the build workflow only).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_03` | joint and conditional probability, Bayes rule, base-rate effect | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/03_probability/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_03
.\build\02_data_science\00_math_foundations\03_probability\s00_math_03.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_03`: a nonzero exit code means a numerical acceptance check failed.

## dsts functions used
None; standard library only.

## Theory
Conditional probability restricts the sample space. Bayes rule reverses conditioning while accounting for the prior; likelihood alone is not posterior probability.

## Math intuition
With prevalence .01, sensitivity .9 and false-positive rate .05, P(disease|positive)=.009/(.009+.0495)=.153846. This is an illustrative probability exercise, not a diagnostic tool.

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code; a nonzero exit means the
mathematical invariant failed.

## Exercise
Vary the prior from .001 to .5 and plot posterior probability; explain base-rate effects.

## Key takeaways
- Posterior is proportional to likelihood times prior; the denominator normalises.
- A rare condition keeps the posterior low even for an accurate test.

## Next module
Continue with **04_statistics** (sampling and uncertainty).
