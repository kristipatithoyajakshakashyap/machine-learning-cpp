# 13_numerical_computing: Floating-Point Precision

## Purpose
Binary floating point cannot represent every decimal, and rounding errors
accumulate in ways that surprise anyone who treats `double` as a real number.
This short module demonstrates catastrophic cancellation, shows how
compensated summation recovers lost precision, and settles on a tolerance
rule for comparing floating-point results. It is the last stop before the
data-science track, where every statistic is a sum of many doubles.

## Prerequisites
Modules 01_basics (types, `double`), 05_file_handling (writing a CSV) and
12_debugging_and_testing (the self-checking lesson pattern used here).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_precision.cpp` | `c13_precision` | Naive vs Neumaier compensated summation on `{1e16, 1, -1e16}`, catastrophic cancellation, tolerance-based equality. Prints nothing and exits non-zero if a check fails | `precision.csv` (both sums, 17 digits) under `results/01_precision_results/` |

The lesson is registered with CTest as `c13_precision`.

## Build and run

```powershell
cmake --preset course                                    # once, from the repo root
cmake --build --preset course --target c13_precision
.\build\01_get_to_know\13_numerical_computing\c13_precision.exe

ctest --preset course -R c13_precision
```

The executable lives under `build/01_get_to_know/13_numerical_computing/`.
On first run it creates `results/01_precision_results/` next to the source
and writes `precision.csv` there (the path is injected as `RUN_OUTPUT_DIR`).

## What the lesson shows

- Addition is not associative: `(1e16 + 1) - 1e16` loses the unit entirely.
- Neumaier compensated summation tracks the low-order rounding residual and
  recovers `1` for this worked example. Both algorithms are O(n) time.
  The compensated version needs only constant extra memory.
- Compare doubles with
  `abs(a - b) <= absolute_tolerance + relative_tolerance * max(abs(a), abs(b))`.
  The absolute term handles values near zero.
- A numerically stable algorithm reduces error but does not replace input and
  domain checks. Avoid `-ffast-math` when testing these guarantees.

Exercises: reverse the input order, compare `float`, `double` and
`long double`, plot the error as `n` small increments are added, and explain
why standardising enormous values overflows a variance calculation.

## Key takeaways
- Never test doubles with `==`. Use an absolute plus relative tolerance.
- Summing many values of different magnitude needs compensation.
- Stable arithmetic and input validation are separate concerns. You need both.

## Next module
You have finished the C++ foundation. Continue with the 02_data_science
track, starting at `02_data_science/00_math_foundations/`, where the `dsts`
library applies everything learned here to real datasets.