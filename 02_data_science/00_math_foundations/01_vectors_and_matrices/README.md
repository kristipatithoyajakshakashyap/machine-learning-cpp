# Vectors and matrices

## Purpose
Introduces the two objects every later lesson manipulates: a vector (one observation or one feature) and a matrix (a whole table). The worked example computes dot products and a matrix-vector product by hand and checks the results against known values.

## Prerequisites
Track 01_get_to_know (loops, `std::vector`, `std::inner_product`). No dsts knowledge needed.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_worked_example.cpp` | `s00_math_01` | dot product, matrix-vector product, dimension checks | `worked_example.csv` under `results/01_worked_example_results/` |

## Build and run

Configure once from the repository root, build the target, then run the
executable from `build/02_data_science/00_math_foundations/01_vectors_and_matrices/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s00_math_01
.\build\02_data_science\00_math_foundations\01_vectors_and_matrices\s00_math_01.exe
```

The lesson links the `dsts` static library and writes its output file to
`results/01_worked_example_results/` next to its source (injected by CMake as
`RUN_OUTPUT_DIR`, created on first run). It is also registered as the ctest
`s00_math_01`: a nonzero exit code means a numerical acceptance check failed.

## dsts functions used
None; plain `std::vector<double>` and the standard library only.

## Theory
A row represents an observation and a column a feature. Dot products combine aligned features; dimension checks prevent accidental broadcasting.

## Math intuition
For x=(1,2,3), w=(2,-1,4), x dot w=12. Matrix-vector multiplication applies this weighted sum independently to each observation.

## Implementation
Read `01_worked_example.cpp`, trace each operation against the equations above,
then run it. Every invariant is asserted in code; a nonzero exit means the
mathematical invariant failed.

## Exercise
Add checked matrix multiplication and test incompatible dimensions.

## Key takeaways
- A dot product is a weighted sum; a matrix-vector product is one dot product per row.
- Always check dimensions before multiplying; silent broadcasting hides bugs.

## Next module
Continue with **02_linear_algebra** (projection and orthogonality).
