# 09_advanced_language — Taking C++ Further

## Purpose
The modern-C++ features that make code shorter, safer, and faster: lambdas,
move semantics, `auto`/`decltype`, `const` correctness, `optional`/`variant`
"maybe" types, threads, and namespaces.

## Prerequisites

Modules 07_stl_core and 08_templates_generics.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_lambdas.cpp` | `conc_lambdas` | Lambdas: functions you write inline, right where you need them | prints only |
| `02_move_semantics.cpp` | `conc_move_semantics` | Move semantics: steal resources instead of copying them | prints only |
| `03_auto_and_decltype.cpp` | `conc_auto_decltype` | auto and decltype: let the compiler name types for you | prints only |
| `04_const_correctness.cpp` | `conc_const_correctness` | const-correctness: promise, with the compiler as watchdog | prints only |
| `05_optional_and_variant.cpp` | `conc_optional_variant` | std::optional and std::variant: honest "maybe" and "one of" types | prints only |
| `06_multithreading.cpp` | `conc_multithreading` | Multiple threads: several functions making progress at once | prints only |
| `07_namespaces.cpp` | `conc_namespaces` | Namespaces: name packages that keep large codebases collision-free | prints only |
| `08_constexpr_and_enum_class.cpp` | `conc_constexpr_enum_class` | Compile-time values with constexpr and type-safe enumerations | prints only |
| `09_mutex_atomic_async.cpp` | `conc_mutex_atomic_async` | Four ways to share work between threads: broken, mutex, atomic, async | prints only |
| `10_function_and_callbacks.cpp` | `conc_function_callbacks` | Passing behaviour around: std::function, lambdas, functors, function | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/09_advanced_language/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_function_callbacks
.\build\01_get_to_know\09_advanced_language\conc_function_callbacks.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** lambdas — inline functions.
- **File 02:** move semantics — steal instead of copy.
- **File 03:** `auto` / `decltype` — compiler-named types.
- **File 04:** `const` correctness — read-only contracts.
- **File 05:** `std::optional` / `std::variant`.
- **File 06:** multiple threads.
- **File 07:** namespaces.
- **File 08:** `constexpr` and `enum class`.
- **File 09:** mutex, atomic, async.
- **File 10:** `std::function`, callbacks, `std::bind`, event handlers.

---

## Files in this folder

### 01_lambdas.cpp
**Key points:** a lambda `[](int x){ return x%2==0; }` is a value you can
pass to `count_if`/`transform`. `[fixed]` captures a **copy**; `[&base]`
captures a **reference** and sees later changes. Named lambdas are stored
in `auto`.
**Expected output:** counts, a value capture, a reference capture that sees
`base` change from 100 to 200, and a transform doubling six numbers.

### 02_move_semantics.cpp
**Key points:** `std::move(x)` makes `x` an rvalue so its resources can be
handed off without a deep copy. A moved-from object is valid but
unspecified (usually emptied). Move-assignment swaps vector ownership in
constant time.
**Expected output:** string and vector moves plus a small
resource-owning `struct` moved by value.

### 03_auto_and_decltype.cpp
**Key points:** `auto` takes the type from the initializer; `decltype(x)`
names the type of an expression without a value; `const auto&` is the
read-only, copy-free loop style.
**Expected output:** each line explains what type `auto`/`decltype`
deduced.

### 04_const_correctness.cpp
**Key points:** `const T&` = read-only (accepts const objects, references,
and temporaries); `T&` = mutable (needs an lvalue); a `const` member
function promises not to mutate `*this`.
**Expected output:** a mutable point scaled, a const point measured, a
temporary measured, and a const range-for.

### 05_optional_and_variant.cpp
**Key points:** `safe_divide` returns `std::optional<double>` — `nullopt`
for a zero divisor, no exceptions needed; `value_or(fallback)` unwraps
safely. `std::variant<int,double,string>` stores exactly one alternative;
`holds_alternative` and `get<T>` read the current one.
**Expected output:** both division outcomes, the fallback, and a variant
viewed while holding `double` then `string`.

### 06_multithreading.cpp
**Key points:** `std::thread(f, args...)` starts a function on a new
thread; `join()` waits for it (never skip it). Shared mutable data is a
race — `std::atomic<int>` fixes counters; per-thread `slots[idx]` avoids
the race entirely.
**Expected output:** four workers each write their own slot, the shared
atomic counter lands exactly on 400000, then main proceeds.

### 07_namespaces.cpp
**Key points:** `namespace math { }` groups names; two namespaces may both
define `title()` and be called `alpha::title()` / `beta::title()`;
`using math::square;` imports one name; a namespace alias like
`namespace fs = std::filesystem;` shortens long names.
**Expected output:** both `title()` calls, `math::square`, `math::average`,
and `square` used after the `using` declaration.

### 08_constexpr_and_enum_class.cpp
**Key points:** `constexpr` functions and variables are evaluated at
compile time when their inputs are constants; `static_assert` proves it.
`if constexpr` inside a template discards the untaken branch, so
`std::to_string` is never instantiated for a `std::string` argument.
`enum class Color : std::uint8_t` never converts to `int` silently and a
`switch` over all enumerators compiles warning-free. Bit-flag enums get
overloaded `|` and `&` so `Perm::Read | Perm::Write` reads naturally.
**Expected output:** `factorial(5) = 120`, the three `describe` lines, the
three traffic-light meanings, `sizeof(Color) = 1`, and the permission checks.

### 09_mutex_atomic_async.cpp
**Key points:** an unsynchronized `++counter` from four threads is a data
race — the printed value is nondeterministic and usually below 400000.
`std::lock_guard<std::mutex>` serialises the critical section;
`std::atomic<int>` makes the increment indivisible without a lock;
`std::async(std::launch::async, f, ...)` returns a `std::future` whose
`get()` waits for the result. All threads are joined before printing.
**Expected output:** the racy value (varies), two exact 400000 counters, and
`sum(1..1000000) = 500000500000` computed as two async halves.

### 10_function_and_callbacks.cpp
**Key points:** a callback is code handed to someone else to run later.
Three callable shapes — function pointers, functors (`operator()`), and
lambdas — all fit in `std::function<R(Args...)>`, a type-erased wrapper, so
one `std::vector<std::function<int(int)>>` can hold a mix. `std::bind`
glues arguments to a callable but a lambda does it more readably. An event
registry is just `map<string, vector<function<void(string)>>>`. Templates
(`auto`/`F`) let the compiler inline the call; `std::function` adds an
indirect call — use templates in hot loops, `std::function` at boundaries.
**Expected output:** `apply` with a function, functor and lambda, a bound
and a lambda `add`, four stored callbacks, `save`/`quit`/`load` events, and
`templated sum = 55, std::function sum = 55`.

---

## Key takeaways
- Lambdas + STL algorithms replace countless hand-written loops.
- `const` everywhere it is true — the compiler enforces it.
- `optional` for "maybe", `variant` for "one of"; `thread` + `atomic` for
  parallelism.
- `std::function` stores any callable; prefer lambdas over `std::bind`.

- Lambdas and `std::function` pass behaviour around; moves steal resources instead of copying.
- `const`, `constexpr` and `enum class` let the compiler enforce your intent.
- Share data between threads only through a mutex, an atomic, or the result of `std::async`.

## Next module

**10_project_structure** - header/source split, include guards, static libraries in CMake.
