# 10_project_structure — Real Programs, Real Folders

## Purpose
Small lessons fit in one file; real programs do not. This module shows the
standard layout that keeps large C++ projects buildable:

```
10_project_structure/
├── include/           # headers (declarations)
│   ├── hello_utils.hpp
│   └── Matrix.hpp
├── src/               # sources (definitions)
│   └── hello_utils.cpp
├── 01_program_layout.cpp
├── 02_include_guards.cpp
└── 03_matrix_library.cpp
```

`src/hello_utils.cpp` is compiled **once** into the `gtk_utils` static
library; every executable links it and `#include`s its header. That is the
exact same pipeline `std::*` comes from.

## Prerequisites

Modules 01-09; in particular classes (03) and templates (08) for the `Matrix` header.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_program_layout.cpp` | `conc_program_layout` | Program layout: how a real project is divided into files | prints only |
| `02_include_guards.cpp` | `conc_include_guards` | Include guards: why every header starts with #ifndef / #define / #endif | prints only |
| `03_matrix_library.cpp` | `conc_matrix_library` | A header-only Matrix class, consumed like any library | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/10_project_structure/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_matrix_library
.\build\01_get_to_know\10_project_structure\conc_matrix_library.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** headers declare, sources define, executables link.
- **File 02:** include guards make headers safe to include twice.
- **File 03:** a header-only `Matrix` class used like a library.

---

## Files in this folder

### include/hello_utils.hpp + src/hello_utils.cpp
**Headers used:** `<string>`.
**Contract:** the header declares `greetings::cheerful(name)` and
`greetings::square(x)`; the source file defines them inside the same
`namespace greetings`. Any `.cpp` sees only the header, so the whole
implementation could change without touching its users.
**CMake:** `add_library(gtk_utils STATIC src/hello_utils.cpp)` builds the
library; `target_include_directories(gtk_utils PUBLIC include)` exports the
folder so linked programs find the headers automatically.

### include/Matrix.hpp
**Key points:** a **header-only** class — every method defined inside the
header. Stored as one flat `std::vector<double>`, element
`data_[r * cols + c]`; `operator()(r, c)` provides the math-y `m(r,c)`
syntax with a const and a non-const overload. Even though a `Matrix(2,3)`
is logically a grid, the flat array is one contiguous block of memory,
which is exactly what fast numerical code wants.

### 01_program_layout.cpp
**Key points:** `#include "hello_utils.hpp"` (quotes = project header, not
library header); calling functions whose bodies live in another
translation unit linked in later. Includes a second header for showdown of
the include path at work.
**Expected output:** the two greeting functions, plus notes about the
header/source/library split.

### 02_include_guards.cpp
**Key points:** includes `hello_utils.hpp` **twice** in one file. The
`#ifndef HELLO_UTILS_HPP / #define / ... / #endif` trio makes the second
include a no-op, so the declarations are seen exactly once and the build
stays clean. Works on every compiler; `#pragma once` is the modern
shortcut, but the guard pattern is what this course uses everywhere.
**Expected output:** a message confirming both includes survived.

### 03_matrix_library.cpp
**Key points:** constructing a `Matrix(2, 3)`, filling it with a double
loop through the write-`operator()`, printing via the read-`operator()`,
summing the flat storage, and writing one element through the reference —
classic library usage with zero `#include` pain.
**Expected output:** the matrix shape, the printed grid `1 2 3 / 4 5 6`,
`element sum = 21`, and the rewritten `(0,0)`.

---

## Key takeaways
- Headers *declare*, sources *define*, libraries *bundle*, executables
  *link*.
- Every header needs an include guard.
- Header-only classes are the easiest libraries of all: users just
  `#include` them.

- Headers declare, sources define; a static library compiles shared code once.
- Include guards (or `#pragma once`) stop a header from being defined twice.
- A header-only template library needs no `.cpp` and is consumed with a single `#include`.

## Next module

**11_capstone** - statistics, CSV dataset reader, normalization, logging and tests combined.
