# 04_error_handling — Failures Are Part of Life

## Purpose
Every real program hits problems: divide by zero, a missing file, text that
is not a number. This module teaches you to handle them **without crashing**.

## Prerequisites

Module 03_oop (classes and inheritance, used to derive custom exception types).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_exceptions.cpp` | `conc_exceptions` | Exceptions: try, throw, catch | prints only |
| `02_custom_exceptions.cpp` | `conc_custom_exceptions` | Custom exceptions: your own error type with extra information | prints only |
| `03_noexcept.cpp` | `conc_noexcept` | noexcept: promising the compiler that a function cannot throw | prints only |
| `04_failures_and_status.cpp` | `conc_failures_and_status` | Two philosophies for handling problems: | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/04_error_handling/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_failures_and_status
.\build\01_get_to_know\04_error_handling\conc_failures_and_status.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** the `try / throw / catch` basics and stack unwinding.
- **File 02:** write your *own* error class with extra details.
- **File 03:** `noexcept` — promises the compiler you cannot throw.
- **File 04:** status codes vs exceptions — *when* to use which.

---

## Files in this folder

### 01_exceptions.cpp — try, throw, catch
**Headers used:** `<iostream>`, `<stdexcept>` (standard error types),
`<string>`.
**Functions:** `divide` (throws `std::runtime_error` on zero),
`grade_at` (throws `std::out_of_range` for bad indexes), a tiny `struct
FileGuard` whose destructor reports cleanup.
**Key points:** `e.what()` gives the message; catching by type; **stack
unwinding** — local objects are destroyed as the exception escapes, so
`FileGuard` cleans up even mid-throw.
**Expected output:** a successful divide, two caught errors, and the unwinding
demo where cleanup prints *before* the catch.

### 02_custom_exceptions.cpp — your own error types
**Headers used:** `<cmath>` (`std::sqrt`), `<iostream>`, `<stdexcept>`,
`<string>`.
**Class/function:** `NegativeValueError : public std::runtime_error` carrying
extra data (`variable()`, `bad_value()`), and `safe_sqrt` which throws it.
**Key points:** derive from `std::exception` family; you can attach any
details you want; catch the **most specific type first**; a `std::exception`
catch swallows every derived error.
**Expected output:** sqrt(9)=3, the custom error caught with its details, and
the same error caught through its base class.

### 03_noexcept.cpp — promising not to throw
**Headers used:** `<iostream>`.
**Functions:** `my_swap` (declared `noexcept`), `safe_divide` (returns a
fallback `0` instead of throwing), `sqrt_guard` (throws).
**Key points:** `noexcept` is a compile-time promise; `static_assert(noexcept(...))`
proves the promise; the commented-out assert shows it *refuses* to compile for
a throwing function.
**Expected output:** the swap, the fallback divide, and three compile-time
check lines (including the "NOT ok" one).

### 04_failures_and_status.cpp — which tool for which problem
**Headers used:** `<iostream>`, `<string>`.
**Function:** `parse_int(const std::string&, int&)` — uses `std::stoi` with a
"how much did I read?" pointer, returns a **bool** and writes the result
through an output reference.
**Key points:** `"12xy"` is not a valid number even though it starts with one;
status codes (bool) suit *expected, everyday* problems, exceptions suit
*unexpected, rare* problems. Real libraries use both.
**Expected output:** ok/123, rejected/abc, rejected/trailing, and the
"status codes vs exceptions" summary line.

---

## The golden rule
> Use **status codes** when the caller is expected to handle it (bad user
> text). Use **exceptions** when it indicates a bug in the program's logic
> (an index that should have been in range).

## Key takeaways
- `throw` hands an error to the nearest matching `catch`; catch by `const&`.
- Derive custom exceptions from `std::runtime_error` and catch the most specific type first.
- Use status codes for expected user errors and exceptions for programming errors.

## Next module

**05_file_handling** - text, binary and CSV files plus `std::filesystem`.
