# 01_basics: The Language Core

## Purpose
Everything you need to write simple, correct, beginner-friendly C++ programs.
No math, no machine learning jargon. Only the language.

## Prerequisites

Module 00_setup (a working compiler and one successful build).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_variables_types.cpp` | `conc_variables_types` | Variables and fundamental data types | prints only |
| `02_operators.cpp` | `conc_operators` | Operators: arithmetic, comparison, logical, bitwise, ternary | prints only |
| `03_control_flow.cpp` | `conc_control_flow` | Control flow: if / else if / else and switch | prints only |
| `04_loops.cpp` | `conc_loops` | Loops: for, while, do-while, range-based for | prints only |
| `05_functions.cpp` | `conc_functions` | Functions: reusable pieces of code | prints only |
| `06_scope_lifetime.cpp` | `conc_scope_lifetime` | Scope and lifetime: where variables live and how long they live | prints only |
| `07_strings.cpp` | `conc_strings` | Text with std::string: joining, measuring, searching, converting | prints only |
| `08_user_input.cpp` | `conc_user_input` | Reading input from the user (std::cin) - and from piped input in tests | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/01_basics/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_user_input
.\build\01_get_to_know\01_basics\conc_user_input.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- Day 1: files 01, 02, 03. Types, operators, decisions.
- Day 2: files 04, 05. Loops and functions.
- Day 3: files 06, 07, 08. Scope, strings, user input.
- After each file, edit it and re-run: change a number, flip a condition,
  add your own print line. Type, don't copy. That is how it sticks.

---

## Files in this folder

### 01_variables_types.cpp: types & sizes
Headers used: `<iostream>` (printing), `<iomanip>` (`std::setprecision` for fixed decimal width), `<type_traits>` (`std::is_same_v` to prove what type `auto` picked).
Key points: `bool`/`char`/`int`/`long`/`float`/`double`, `sizeof()`, `auto`. Integer division drops the fraction. `static_cast<double>` forces decimal math. `float` vs `double` precision.
Expected output: a table of byte sizes, then precision of `3.14159265` (`float` → `3.14159274`, `double` → `3.14159265`), and a `long` storing `285000`.

### 02_operators.cpp: operators
Headers used: `<iostream>`.
Key points: arithmetic (`+ - * / %`), comparisons, `&& || !`, bitwise (`& | ^ ~ << >>`), and the ternary `condition ? a : b`. `std::boolalpha` makes bools print as `true`/`false`.
Expected output: sums, `7/2 == 3` (int) vs `3.5` (double), comparison and logic results, bit tricks (`5 & 3 = 1`, `1 << 4 = 16`), a passing-grade check and a max-of-two using `? :`.

### 03_control_flow.cpp: if / else if / else and switch
Headers used: `<iostream>`, `<string>`.
Functions: `fizzbuzz(int n)` (classic FizzBuzz), `categorize(int n)` (positive/zero/negative).
Key points: `if/else if/else` chains, `switch` with `break` and `default`, and overloading consecutive `if` checks via early `return`.
Expected output: letter grades for 85 and 55, day names for 3/7/9, FizzBuzz for 5/3/15/7, and `categorize(7), (0), (-5)`.

### 04_loops.cpp: loops
Headers used: `<iostream>`, `<vector>` (`std::vector`, a growable list).
Key points: `for`, nested `for` (addition table), `while` (countdown), `do-while` (runs at least once), range-based `for`, `continue`.
Also learned the hard way: never test a `double` against exactly `0.0`. A shrink-and-loop stalls forever at the smallest positive number. Always compare against a sensible limit like `1.0`.
Expected output: a count, a 3x3 addition table, countdown `5 4 3 2 1`, "halved 7 times", three payments clearing a debt, `10 20 30 40`, multiples of 3 skipped, and `sum 1..10 = 55`.

### 05_functions.cpp: functions
Headers used: `<iostream>`, `<string>`.
Functions: `add`, `greet`, `multiply` (two overloads: int and double), `volume` (default argument), `weather` (early return), `is_even` (returns bool).
Key points: return types, passing `const std::string&` to avoid copying, default arguments, overloading, early return.
Expected output: arithmetic results, greetings, `10`/`24`, volume with and without default height, weather for 32/22/5, and `true`/`false` from `is_even`.

### 06_scope_lifetime.cpp: scope & lifetime
Headers used: `<iostream>`.
Functions: `next_call()` demonstrating a `static` local variable.
Key points: variables live only inside their `{ }` block. Globals are visible below their declaration. Shadowing is possible but messy. `static` locals remember their value between calls. Loop counters survive the loop.
Expected output: global = 10, sum 1..5 = 15, an inner `secret`, a shadowed value that reverts, static calls `1 2 3`, and loop counter `i = 3` afterwards.

### 07_strings.cpp: std::string
Headers used: `<iostream>`, `<string>`.
Key points: `+` concatenation, `.size()`, `[i]`/`.at(i)` character access, `.find()`, `.substr()`, `std::to_string`, `std::stod`, `==`, `.empty()`.
`to_string(3.14)` always writes 6 decimals: `"3.140000"`.
Expected output: the joined string and its length (16), first/`position 3` characters, `'under'` found at position 11, `'morning'` extracted, `"42"`/`"3.140000"`/`3.14` conversions, apple==apple true, empty string size 0.

### 08_user_input.cpp: reading input
Headers used: `<iostream>`, `<string>`.
Key points: `std::getline` reads a whole line, `std::cin >> x` reads one token, the leftover-newline pitfall when mixing them, `if (!std::cin)` catches bad input needing `clear()` + `ignore()`, and a `while` loop re-asks until input is valid (an age 1–120).
To test non-interactively (PowerShell):
```powershell
"Alice`n42`nthe matrix is cool`n25" | .\build\01_basics\conc_user_input.exe
```
Expected output: greeting, `42 doubled is 84`, the whole sentence echoed, and `Your age 25 is saved.`

---

## What to do next
You now have every tool to write small programs. Module 02 teaches the memory
behind them: arrays, pointers, references and smart pointers. These are the
core of writing fast, safe C++.

## Key takeaways
- Types have sizes. Integer and floating-point division behave differently.
- `if`/`switch`, `for`/`while`/`do-while` and functions are the whole control vocabulary.
- Pass strings as `const std::string&`. Validate `std::cin` before trusting input.

## Next module

02_memory_management - arrays, vectors, pointers, references and smart pointers.