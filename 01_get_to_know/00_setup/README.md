# 00_setup — Your First C++ Program

## Purpose
This folder gets your machine ready and runs the very first program.

## Prerequisites

A C++17 compiler (the course uses MinGW g++ from MSYS2 UCRT64) and CMake 3.16+. No prior C++ knowledge.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `hello.cpp` | `hello` | Your very first C++ program | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/00_setup/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target hello
.\build\01_get_to_know\00_setup\hello.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Building and running
```powershell
# whole project (from the 01_get_to_know folder)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# run this lesson's program
.\build\00_setup\hello.exe
```

Or compile the one file directly:
```powershell
g++ hello.cpp -std=c++17 -Wall -Wextra -Werror -pedantic -o hello.exe
.\hello.exe
```

## Files in this folder

### CMakeLists.txt
Tells CMake how to turn `hello.cpp` into an executable named `hello`.
The `add_lesson_executable(...)` helper is defined in the root `CMakeLists.txt`
and applies to every lesson folder.

### hello.cpp
| Part of the code | What it does |
|------------------|--------------|
| `#include <iostream>` | Loads the **input/output stream** library so we can print text with `std::cout`. |
| `int main()` | The **entry point**. Every program starts executing here. It must return an `int` — `0` means "everything went fine". |
| `std::cout << "..." << std::endl` | Prints text to the console. `std::endl` ends the line. |
| `__cplusplus` | A magic number the compiler fills in: `201703` literally means **C++17**. |

## Expected output
```
Hello, Machine Learning world!
We are using C++ version: 201703
```

## Why this matters for C++
Everything after this lesson is just more `main()` functions that use
variables, loops, classes, and files — but the skeleton you see here
never changes.

## Key takeaways
- Every C++ program is a `main()` that returns an exit code; `0` means success.
- `#include <iostream>` gives you `std::cout` for printing.
- The build is driven by CMake presets; you never compile by hand once configured.

## Next module

**01_basics** - variables, operators, control flow, loops, functions, strings and user input.
