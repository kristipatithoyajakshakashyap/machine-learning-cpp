# 01_get_to_know — Zero-to-Advanced C++ for Machine Learning

## Purpose
This track builds the C++ foundation used by the data-science and
machine-learning tracks that follow. It starts from "hello world" and ramps
up to templates, move semantics, threads, project layout, contracts and
floating-point numerics. Every lesson is a standalone `.cpp` that compiles
with `-Wall -Wextra -Werror -pedantic` (zero warnings allowed) and prints its
expected output in a header comment, so you can always check yourself.

- Standard: **C++17**
- Compiler: the course uses MinGW g++ (`D:/msys64/ucrt64/bin/g++.exe`);
  any C++17-capable GCC/Clang works
- Build system: CMake 3.16+ driven by the `course` preset in
  `CMakePresets.json` at the repository root

## Prerequisites
None beyond a working toolchain. Module 00_setup checks that one build
succeeds before anything else.

## Modules

| Module | Topic | What you will master | Writes files? |
|--------|-------|----------------------|---------------|
| [00_setup](00_setup/README.md) | Toolchain | Compiling C++, first program, working with CMake | no |
| [01_basics](01_basics/README.md) | Language core | Types, operators, control flow, loops, functions, scope, strings, user input | no |
| [02_memory_management](02_memory_management/README.md) | Memory | Arrays, vectors, pointers, references, new/delete, smart pointers | no |
| [03_oop](03_oop/README.md) | Object oriented | Classes, constructors, encapsulation, inheritance, polymorphism, operators, Rule of Five | no |
| [04_error_handling](04_error_handling/README.md) | Robustness | Exceptions, custom exception types, `noexcept`, status-vs-exception patterns | no |
| [05_file_handling](05_file_handling/README.md) | I/O | Text, binary, CSV parsing, `std::filesystem` | yes |
| [06_logging](06_logging/README.md) | Observability | Log levels, timestamps, file logging, a reusable `Logger` class | yes |
| [07_stl_core](07_stl_core/README.md) | Standard library | Containers, iterators, algorithms, `<random>`, `<chrono>`, `<regex>` | no |
| [08_templates_generics](08_templates_generics/README.md) | Generics | Function/class templates, specialization, variadics, type traits | no |
| [09_advanced_language](09_advanced_language/README.md) | Modern C++ | Lambdas, move semantics, `auto`/`decltype`, `constexpr`, `optional`/`variant`, threads, callbacks | no |
| [10_project_structure](10_project_structure/README.md) | Architecture | Header/source split, include guards, header-only `Matrix<T>`, static libraries in CMake | no |
| [11_capstone](11_capstone/README.md) | Graduation | Statistics, CSV dataset reader, normalization, logger integration, a test harness | yes |
| [12_debugging_and_testing](12_debugging_and_testing/README.md) | Correctness | Contracts, sanitizer/debugger recipe, a ~40-line test framework (CTest `c12_*`) | no |
| [13_numerical_computing](13_numerical_computing/README.md) | Numerics | Catastrophic cancellation, compensated summation, tolerances (CTest `c13_precision`) | yes |

Each module README lists every lesson with its file, build target, what it
teaches and what it writes.

## Build and run

Configure once from the repository root, then build either everything or a
single lesson target. Executables land under
`build/01_get_to_know/<module>/`.

```powershell
cmake --preset course                                        # configure (once)
cmake --build --preset course                                # build every lesson
cmake --build --preset course --target conc_variables_types  # or just one
.\build\01_get_to_know\01_basics\conc_variables_types.exe
```

Lesson targets are named `conc_<topic>` for modules 00-11 (plus `hello` in
00_setup) and `c12_*` / `c13_*` for the last two modules; the exact names are
in each module's `CMakeLists.txt` and README.

### Runtime artifacts
Lessons that write files (CSV, logs, binary data) write into the owning
module's `results/<source stem>_results/` directory, for example
`05_file_handling/03_csv_files.cpp` writes to
`05_file_handling/results/03_csv_files_results/`. The absolute path is
injected into every executable through the `RUN_OUTPUT_DIR` compile
definition and the directory is created lazily on first run.

## Verification
The preset configures with `BUILD_TESTING=ON`. Run the C++ track checks with

```powershell
ctest --preset course -R "c12_|c13_|cpp_normalizer"
```

They cover checked-mean contracts, the bounded buffer, the test harness,
numerical compensation, and the actual capstone normalizer (`tests/`) for
empty, constant and non-finite inputs. Expected-output comments in the
lessons are teaching examples, not an assertion that every historical program
was re-executed in the current revision.

## Key takeaways
- Compile clean under `-Werror`; warnings are bugs you have not met yet.
- Ownership is explicit: references and smart pointers over raw `new`/`delete`.
- Errors are values or exceptions, never silently ignored return codes.
- Split declarations from definitions once a program outgrows one file.
- Tests must fail through their exit code; `assert` disappears in Release.

## Next
Continue with the sibling **02_data_science** track (the `dsts` library and
its lessons) and then **03_ml_course**. See `../docs/learning_path.md` for the
study order.
