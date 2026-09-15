# 02_memory_management — Arrays, Vectors, Pointers, Smart Pointers

## Purpose
C++ gives you two memories to live in: the **stack** (fast, small, automatic)
and the **heap** (big, manual). This module is about storing lots of values
and managing memory correctly — the foundation of every fast program.

## Prerequisites

Module 01_basics (variables, loops, functions).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_arrays_vectors.cpp` | `conc_arrays_vectors` | Storing many values: C-style arrays and std::vector | prints only |
| `02_pointers.cpp` | `conc_pointers` | Pointers: variables that store the memory ADDRESS of another thing | prints only |
| `03_references.cpp` | `conc_references` | References: a second name (alias) for a variable that already exists | prints only |
| `04_dynamic_memory.cpp` | `conc_dynamic_memory` | Dynamic memory: asking for memory while the program is running | prints only |
| `05_smart_pointers.cpp` | `conc_smart_pointers` | Smart pointers: let the program delete memory for you (no new/delete) | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/02_memory_management/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_smart_pointers
.\build\01_get_to_know\02_memory_management\conc_smart_pointers.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **Files 01 → 02:** arrays/vectors, then pointers (the hardest idea in C++ —
  read slowly, re-run, trace `&x`/`*p` by hand on paper).
- **File 03:** references — they are pointers made easy.
- **Files 04 → 05:** manual `new`/`delete`, then smart pointers (never touch
  `new`/`delete` by hand again).

---

## Files in this folder

### 01_arrays_vectors.cpp — C-style arrays and std::vector
**Headers used:** `<iostream>`, `<vector>`.
**Functions/key points:** a fixed-size C array `int scores[5]`; a growing
`std::vector<int> grades` with `push_back`, `.size()`, `.capacity()`,
`[i]`/`.at(i)`; a loop finding the maximum; a 2D `int grid[3][3]` printed as a
TicTacToe board.
**Expected output:** scores, growing vector (size 3, capacity 4 here),
`high`est grade 90, the 3x3 board and its `[1][2]` cell.

### 02_pointers.cpp — pointers
**Headers used:** `<iostream>`.
**Functions/key points:** `&x` = address of `x`; `int* p` stores an address;
`*p` reads/writes through it; `nullptr` for "points nowhere" (test before
use); pointer arithmetic `*(p + i)` walks an array; a function `add_ten`
changes `main`'s variable via its address.
**Expected output:** reading 42 through a pointer, writing 99 into `x`,
a safe null check, array `10 20 30 40`, `*(p + 2) = 30`, then `x = 109`.

### 03_references.cpp — references
**Headers used:** `<iostream>`, `<string>`.
**Functions/key points:** `int& alias = a` makes an alias that IS `a`;
`swap_ints(int& x, int& y)` swaps the caller's variables; a `const std::string&`
parameter reads without copying; a reference cannot be re-attached.
**Expected output:** `a is now 20 (set through its alias)`, swap `3 7 → 7 3`,
string length 17, and name/nickname agreement.

### 04_dynamic_memory.cpp — new and delete
**Headers used:** `<iostream>`.
**Key points:** `new int(7)` creates heap memory and returns its address;
`delete` frees ONE value, `delete[]` frees an ARRAY; after deleting set the
pointer to `nullptr`; the stack/heap mental model and four rules of thumb are
explained in the comment block under `main`.
**Expected output:** heap int = 7, heap array `10 20 30 40 50`, and a note
that pointers were reset to `nullptr`.

### 05_smart_pointers.cpp — unique_ptr / shared_ptr / weak_ptr
**Headers used:** `<iostream>`, `<memory>`, `<string>`.
**Functions/key points:** a small `struct Toy` that reports its creation and
destruction; `std::make_unique` (one owner), `std::make_shared` (many owners,
freed when the last one dies), `std::weak_ptr` (a watcher that does not keep
things alive); `use_count()` shows the number of owners; `lock()` re-grabs a
checked pointer.
**Expected output:** unique 5, shared 10 with use-count 2 → 1, weak lock while
alive then empty after reset, and `Toy` printed as created-only-once and
destroyed automatically.

---

## Key takeaways
1. **Prefer `std::vector`** over raw arrays.
2. **Prefer smart pointers** over raw `new`/`delete`.
3. Use references when you just want an alias; pointers when you need to
   point elsewhere, to be nullable, or to walk memory step by step.

- `std::vector` grows at runtime; C-style arrays do not.
- A pointer stores an address, a reference is an alias; both let functions modify caller data.
- Prefer `unique_ptr`/`shared_ptr` from `make_unique`/`make_shared` over raw `new`/`delete`.

## Next module

**03_oop** - classes, constructors, encapsulation, inheritance, polymorphism, operators.
