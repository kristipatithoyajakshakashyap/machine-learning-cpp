# 08_templates_generics: One Definition, Many Types

## Purpose
Templates let you write logic ONCE for every type instead of copy-pasting
for `int`, `double`, `std::string`, ... The compiler stamps out a concrete
version for each type you use. This is why `std::vector<int>` and
`std::vector<double>` work with the same library code.

## Prerequisites

Module 07_stl_core (using templated containers before writing your own).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_function_templates.cpp` | `conc_function_templates` | Function templates: ONE function definition, MANY types | prints only |
| `02_class_templates.cpp` | `conc_class_templates` | Class templates: one class instantiated for many types | prints only |
| `03_specialization.cpp` | `conc_specialization` | Specialization: giving a template SPECIAL treatment for one type | prints only |
| `04_variadic_templates.cpp` | `conc_variadic_templates` | Variadic templates: ONE function that accepts any number of arguments | prints only |
| `05_type_traits_sfinae.cpp` | `conc_type_traits_sfinae` | Asking the compiler questions about types: <type_traits>, SFINAE, the detection idiom | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/08_templates_generics/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_type_traits_sfinae
.\build\01_get_to_know\08_templates_generics\conc_type_traits_sfinae.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- File 01: function templates. `max_of`, `show`.
- File 02: class templates. `Box<T>`, `PairBox<A,B>`.
- File 03: specialization. Special-casing certain types.
- File 04: variadic templates. Any number of arguments, fold expressions.
- File 05: type traits, SFINAE, the detection idiom, `if constexpr`.

---

## Files in this folder

### 01_function_templates.cpp
Key points: `template <typename T>` starts every definition, `T const&` avoids copying, and the compiler generates a separate function per used type. A function template must be visible before the call site. String literals are `const char[N]` of different lengths, so `max_of("pear","peach")` needs `std::string` wrappers to give both arguments one matching `T`.
Expected output: int, double, and string comparisons plus two `show(vector<T>)` calls.

### 02_class_templates.cpp
Key points: `Box<int>`, `Box<double>`, `Box<std::string>` are three distinct types built from one blueprint. Getters/setters (`get()`/`set()`) are the safe read/change interface. `PairBox<First, Second>` shows a class with two template parameters, like `std::pair`.
Expected output: the three boxes, a `set()` update, and a printed pair.

### 03_specialization.cpp
Key points: the primary template is the default (`"unknown"`). A full specialization `template <> struct TypeName<int>` overrides exactly one type. A partial specialization `TypeName<T*>` matches any pointer and delegates to `TypeName<T>` so `double**` resolves recursively. Function templates cannot be partially specialized. That pattern needs a class template.
Expected output: `int`, `double`, a fallback `unknown`, and nested pointer names.

### 04_variadic_templates.cpp
Key points: `...` captures "the rest of the arguments". The classic style recurses: one overload handles a single argument (base case) and a variadic overload peels one argument off each call. The C++17 fold expression `(values + ... + 0)` sums a whole pack in one line and `(parts + ...)` concatenates strings. `print_args` shows a mixed-type pack. Passing `1.5` to an `int` parameter silently chops it to `1`.
Expected output: recursive and folded sums, joined strings, and a mixed-type print.

### 05_type_traits_sfinae.cpp
Key points: `<type_traits>` answers compile-time questions: `is_same_v`, `is_integral_v`, `is_floating_point_v`. `decay_t` strips references/const/arrays. `conditional_t` picks a type from a boolean. `enable_if_t` in a return type removes an overload instead of erroring (SFINAE), so an integral and a floating `halve` coexist. `std::void_t` builds a `has_size<T>` detector that is true exactly when `t.size()` compiles. `if constexpr` does the same job inside one function and is the clearer C++17 choice. `static_assert` documents every answer.
Expected output: trait true/false lines, `halve(7) -> 3` vs `halve(7.0) -> 3.5`, `has_size` for vector/string/int, `length` of three inputs, and `describe` classifying int/double/string.

---

## Key takeaways
- `template <typename T>` = for every T. The compiler stamps out copies.
- Full specializations target one concrete type. Partial specializations
  target a whole shape like `T*`.
- Variadic packs + fold expressions handle "any number of arguments".
- Type traits ask questions at compile time. Prefer `if constexpr` over
  SFINAE when one function body holds every case.

- A template is one definition instantiated per type. The compiler generates the code.
- Specialization gives one type special treatment. Variadics accept any number of arguments.
- `if constexpr` and `<type_traits>` replace most SFINAE tricks in C++17.

## Next module

09_advanced_language - lambdas, move semantics, `constexpr`, `optional`/`variant`, threads.