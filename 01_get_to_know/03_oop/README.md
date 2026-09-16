# 03_oop: Object-Oriented Programming

## Purpose
Objects bundle data (fields) with the functions (methods) that use
them. This module builds up the full object-oriented toolkit: classes,
constructors, encapsulation, inheritance, polymorphism, and the memory rules
of the Rule of Five.

## Prerequisites

Modules 01_basics and 02_memory_management (functions, references, dynamic memory).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_classes_objects.cpp` | `conc_classes_objects` | Classes and objects: bundling data with the functions that use it | prints only |
| `02_constructors.cpp` | `conc_constructors` | Constructors: the special functions that build an object | prints only |
| `03_encapsulation.cpp` | `conc_encapsulation` | Encapsulation: hiding data and guarding how it changes | prints only |
| `04_inheritance.cpp` | `conc_inheritance` | Inheritance: building a new class on top of an existing one | prints only |
| `05_polymorphism.cpp` | `conc_polymorphism` | Polymorphism: one interface, many behaviors | prints only |
| `06_operator_overloading.cpp` | `conc_operator_overloading` | Operator overloading: teaching +, ==, and << to your own classes | prints only |
| `07_rule_of_five.cpp` | `conc_rule_of_five` | The Rule of Five: when your class owns memory, you must provide all five special members | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/03_oop/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_rule_of_five
.\build\01_get_to_know\03_oop\conc_rule_of_five.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- Files 01 → 02: what a class is, and how objects are born (constructors).
- File 03: encapsulation. Hiding data and guarding how it changes.
- Files 04 → 05: inheritance (IS-A) and polymorphism (one call, many
  behaviors).
- Files 06 → 07: overloading operators and the Rule of Five. That is the mental
  model behind every well-behaved class.

Read each file's `EXPECTED OUTPUT` first, run the program, and check you
understand why each line appears. Then change one thing and re-run.

---

## Files in this folder

### 01_classes_objects.cpp: classes, objects, access
Headers used: `<iostream>`, `<string>`.
Types/functions: `class Student` with `public` members and a `print()` method, `struct Point` (members public by default), and `struct StudentSecret` showing that `class` defaults to private.
Key points: a class is a blueprint, an object is one instance, public vs private, and `const` methods (marked `const` so const objects call them).
Expected output: two students printed, a working `p.x = 3.5`, and a note that the commented-out private write would not compile.

### 02_constructors.cpp: constructors
Headers used: `<iostream>`, `<string>`.
Functions: `Student()` default, `Student(name, id)` initializer list, `Student(const Student&)` copy constructor, `print()`, `rename()`.
Key points: constructors run automatically at object birth. Initializer lists fill members in order. The copy constructor creates independent copies (changing the copy does not change the original).
Expected output: default `unknown (0)`, `Ada (101)`, a copy `Ada (101)`, and the original still `Ada (101)` after renaming the copy.

### 03_encapsulation.cpp: encapsulation (gating data)
Headers used: `<iostream>`, `<string>`.
Functions: `Person(name)`, `get_age()`, `set_age(int)`.
Key points: age is `private`. The setter is the only way to modify age and it validates the range `0..130`. Impossible values like `-5` or `999` are refused, so the class invariant, age is always sensible, always holds.
Expected output: age 0 → set to 25 → refused -5 (stays 25) → refused 999 (stays 25).

### 04_inheritance.cpp: inheritance (IS-A)
Headers used: `<iostream>`, `<string>`.
Classes/functions: `Animal` base (constructor, `describe()`, virtual `speak()`), `Dog`/`Cat` derive from it, call `Animal(n)` in their own constructor and `override speak()`.
Key points: `class Dog : public Animal` means a Dog is an Animal, `protected` members are visible to base and derived classes, `override` replaces base behavior, and `virtual` selects the real version at runtime.
Expected output: `Rex says Woof!`, `Tom says Meow!`.

### 05_polymorphism.cpp: polymorphism (one interface, many shapes)
Headers used: `<cmath>` (`std::acos(-1.0)` to get pi), `<iomanip>` (`std::setprecision`), `<iostream>`, `<memory>` (`unique_ptr`), `<vector>`.
Classes/functions: abstract `Shape` with pure virtual `area() = 0`, `Circle::area()` and `Rectangle::area()` implement the same name differently, and `virtual ~Shape()` ensures cleanup through a base pointer.
Key points: you cannot instantiate an abstract class. The same call `shape->area()` picks the correct math at runtime. That is polymorphism. A `std::vector<std::unique_ptr<Shape>>` holds different shapes uniformly.
Expected output: `Circle(radius 3) = 28.2743`, `Rectangle(4 x 5) = 20.0000`.

### 06_operator_overloading.cpp: operators for your own types
Headers used: `<iostream>`.
Functions: member `operator+=`, free `operator+`, `operator<<`, `operator==`, and `friend` declarations grant them private access.
Key points: operators are functions with special names. `+` reuses `+=` so the math lives in one place. `operator<<` teaches `std::cout` how to print your object. `operator==` makes comparisons work.
Expected output: vectors printed as `(1, 2)`, `(3, 4)`, sum `(4, 6)`, after `+=` still `(4, 6)`, equality true.

### 07_rule_of_five.cpp: managing owned memory
Headers used: `<algorithm>` (`std::copy`), `<cstddef>` (`std::size_t`), `<iostream>`, `<utility>` (`std::move`).
Class: `SmartBuffer` owns an `int[]` and implements the full five special members: destructor, copy constructor, copy assignment, move constructor, move assignment. It also has static counters `copies_made`/`moves_made`.
Key points: copying duplicates memory (independent objects). Moving steals memory (source becomes empty, cheap). Every `new` must have a `delete` (or use smart pointers). Self-assignment guard `this != &other`.
Expected output: tracked counts growing 1→1→2→2 copies and 0→1→1→2 moves, with A emptied after moving from it.

---

## Key takeaways
- Encapsulation hides data. Inheritance expresses IS-A. Polymorphism sends one call to the right behavior.
- A class that manages memory needs the Rule of Five, or use smart pointers.
- Prefer const-correct, minimal interfaces. Declare only what your class needs.

- A class bundles data with the functions that keep it valid. Keep fields private.
- Virtual functions let one interface dispatch to many implementations at runtime.
- A class that owns memory needs the Rule of Five (destructor, copy/move ctor and assignment).

## Next module

04_error_handling - exceptions, custom exception types, `noexcept`, status codes.