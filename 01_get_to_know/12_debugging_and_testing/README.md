# 12_debugging_and_testing — Contracts, Sanitizers and a Test Harness

## Purpose
Correct programs need more than a clean compile. This module shows how to
state what a function requires (its *contract*), how to make violations loud
instead of silent, how to run the code under a debugger and under sanitizers,
and how to write a tiny test framework whose exit code CTest can trust.
Every lesson here is also registered as a CTest test, so `ctest` re-runs the
same checks you read in the source.

## Prerequisites
Modules 04_error_handling (exceptions), 09_advanced_language (lambdas,
`constexpr`) and 11_capstone (the self-built PASS/FAIL harness this module
generalises).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_contracts.cpp` | `c12_contracts` | `checked_mean` with input contracts (empty and non-finite input rejected), verified by a self-test | prints only |
| `02_sanitizers_and_debugger.cpp` | `c12_sanitizers` | A `BoundedBuffer` whose `push`/`pop`/`at` throw on contract violations, plus a printed recipe for `-fsanitize=address,undefined` and gdb | prints only |
| `03_test_harness.cpp` | `c12_test_harness` | A ~40-line test framework (`TEST`, `EXPECT_TRUE`, `EXPECT_EQ`, `EXPECT_NEAR`, file:line failure reports) exercising a Welford `RunningStats` struct | prints only |

CTest names match the targets: `c12_contracts`, `c12_sanitizers`,
`c12_test_harness`. Each executable prints `N passed, M failed` (or the
equivalent) and exits non-zero when anything failed.

## Build and run

```powershell
cmake --preset course                                    # once, from the repo root
cmake --build --preset course --target c12_contracts
.\build\01_get_to_know\12_debugging_and_testing\c12_contracts.exe

# run the three lessons as tests
ctest --preset course -R c12_
```

Executables live under `build/01_get_to_know/12_debugging_and_testing/`.

## How to work through it

1. **Debugger.** Build `c12_contracts` in a Debug configuration, set a
   breakpoint in `checked_mean`, inspect the vector and step through the
   incremental mean update. Step into the empty-input case and watch the
   exception being thrown; then run CTest to reproduce the same behaviour
   without the debugger.
2. **What to test.** Test normal behaviour, boundaries, rejected input and
   mathematical properties. Tests must return non-zero on failure in Release
   too; `assert` alone disappears under `NDEBUG`. Translation invariance
   (shifting every input shifts the mean by the same amount) is a useful
   property beyond a single hard-coded expected output. Floating-point
   comparisons need a tolerance (`EXPECT_NEAR`).
3. **Complexity.** One traversal gives O(n) time and O(1) extra storage.
4. **Sanitizers.** With GCC/Clang on supported platforms, build a separate
   diagnostic configuration with `-fsanitize=address,undefined` and
   `-fno-omit-frame-pointer`; do not claim sanitizer coverage until it has
   actually run. Lesson 02 prints the exact commands.

Exercises: add non-finite rejection tests, compare a two-pass mean with the
incremental one, and investigate overflow at extreme magnitudes.

## Key takeaways
- A contract violation should throw (or abort) - never return a plausible
  wrong number.
- `assert` vanishes in Release builds; tests must fail through their exit code.
- A test framework is a registry of functions plus a few `EXPECT_*` macros;
  you do not need a library to get file:line failure reports.
- Sanitizers and a debugger find the bugs tests only hint at.

## Next module
**13_numerical_computing** - floating-point precision, compensated summation
and tolerance-based comparison.
