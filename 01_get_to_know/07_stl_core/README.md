# 07_stl_core — The Standard Library in Everyday Use

## Purpose
The Standard Template Library (STL) ships storage and logic with the
compiler. This module learns the five core containers, the iterators that
traverse them, and the `<algorithm>` functions that sort, count, and search.
From here on every later module leans on these building blocks.

## Prerequisites

Modules 01-03 (loops, vectors, classes). Lambdas from 09 are previewed but not required.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_containers_overview.cpp` | `conc_containers_overview` | The Standard Template Library: the ready-made containers you build on | prints only |
| `02_vector_and_list.cpp` | `conc_vector_and_list` | vector vs list: where each one is fastest | prints only |
| `03_stack_and_queue.cpp` | `conc_stack_and_queue` | Stack and queue: two strict orderings that organize work | prints only |
| `04_set_and_map.cpp` | `conc_set_and_map` | set and map: sorted, fast, deduplicated lookups | prints only |
| `05_iterators.cpp` | `conc_iterators` | Iterators: the uniform way to walk through any container | prints only |
| `06_algorithms.cpp` | `conc_algorithms` | Algorithm headers: ready-made logic for sorting, searching, counting | prints only |
| `07_array_and_string_view.cpp` | `conc_array_string_view` | std::array (fixed size, on the stack) and std::string_view (no-copy text) | prints only |
| `08_random_and_chrono.cpp` | `conc_random_and_chrono` | Reproducible random numbers with <random> and wall-clock timing with | prints only |
| `09_regex.cpp` | `conc_regex` | Pattern matching with <regex>: match, search, replace, capture groups, | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/07_stl_core/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_regex
.\build\01_get_to_know\07_stl_core\conc_regex.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** meet the five containers and when to pick each.
- **File 02:** vector vs list — where each is fast.
- **File 03:** stack (LIFO) and queue (FIFO) orderings.
- **File 04:** set and map — sorted, unique, fast lookups.
- **File 05:** iterators — the common way to walk any container.
- **File 06:** the `<algorithm>` toolbox.
- **File 07:** `std::array` and `std::string_view`.
- **File 08:** `<random>` and `<chrono>` — seeded randomness and timing.
- **File 09:** `<regex>` — pattern matching and tiny parsers.

---

## Files in this folder

### 01_containers_overview.cpp — the five containers
**Headers used:** `<deque>`, `<list>`, `<map>`, `<set>`, `<string>`,
`<vector>`.
**Template:** `show(label, note, container)` prints any container's elements
in `[a b c]` style (this is the first taste of a function template — module
08 explains them fully).
**Key points:** vector = contiguous, fast at the back; list = linked nodes,
fast inserts anywhere; deque = fast at both ends; set = sorted + unique;
map = sorted key→value table. The map is printed separately because its
elements are `std::pair`s.
**Expected output:** one bracketed line per container plus a sizes summary.

### 02_vector_and_list.cpp — where each is fast
**Headers used:** `<list>`, `<vector>`.
**Template:** `print(label, container)`.
**Key points:** `push_back` on a vector is cheap; `insert(begin(), x)`
shifts every element; `list` makes `push_front` cheap. `insert`/`erase` take
iterator positions (`v.begin() + 1`).
**Expected output:** vector after pushes/insert/erase and a list after
front+back pushes, with sizes.

### 03_stack_and_queue.cpp — two orderings
**Headers used:** `<deque>`, `<queue>`, `<stack>`.
**Key points:** stack pops `top()` first (LIFO); queue pops `front()` first
(FIFO); a deque lets you push both ends. `while (!c.empty())` is the
standard drain loop.
**Expected output:** stack pops `3 2 1`, queue pops `1 2 3`, deque shows
`[9 1 2 3 5]`.

### 04_set_and_map.cpp — sorted lookups
**Headers used:** `<map>`, `<set>`, `<string>`.
**Key points:** a set silently merges duplicates and stays sorted —
iterating always comes out `1 3 5 7 9`; `count(k)` asks "is it there?";
map keeps `key` sorted and `operator[]` can read or write; `find` returns an
iterator (compare with `end()`).
**Expected output:** deduplicated set, two counts, map printed in sorted key
order, an update, and a successful `find`.

### 05_iterators.cpp — walking anything
**Headers used:** `<iterator>` (`distance`), `<vector>`.
**Key points:** `begin()`/`end()` are positions; `*it` reads/writes;
`+=` jumps; `rbegin()/rend()` walk backwards; range-for is sugar over
iterators; `std::distance` measures a range.
**Expected output:** first element, a +2 jump, forward and reverse walks,
values doubled through the iterator, and `distance(begin,end) = 4`.

### 06_algorithms.cpp — the algorithm toolbox
**Headers used:** `<algorithm>`, `<numeric>` (`accumulate`).
**Functions:** `sort`, `min_element`/`max_element`, `count_if` (with a
lambda predicate), `accumulate`, `find`, `transform` (with an `*2`
lambda).
**Key points:** lambdas are tiny inline functions passed to algorithms —
module 09 covers them in detail. `find` returns an iterator; subtracting
`begin()` yields an index.
**Expected output:** the vector shown before/during/after each algorithm.

### 07_array_and_string_view.cpp — fixed arrays and text views
**Headers used:** `<array>`, `<string_view>`.
**Key points:** `std::array<T,N>` is a stack-allocated aggregate with a
compile-time size that works in `constexpr` code and unpacks with
structured bindings. `std::string_view` is a pointer+length view over
existing text: `substr`/`find` never copy, and any function taking a view
accepts `std::string`, literals, and other views. Never let a view outlive
its owner (the dangling pitfall is shown in comments only).
**Expected output:** array size/elements, bindings, a compile-time sum,
view length/substr/find, vowel counts, and a comparison table of
array / vector / string / string_view.

### 08_random_and_chrono.cpp — reproducible randomness and timing
**Headers used:** `<random>`, `<chrono>`, `<algorithm>` (`shuffle`).
**Key points:** `std::mt19937` is a seeded engine; `uniform_int_distribution`,
`uniform_real_distribution`, `normal_distribution` and `bernoulli_distribution`
shape its bits. The same seed always replays the same sequence, and
`std::shuffle(begin, end, engine)` makes permutations reproducible too.
`steady_clock::now()` differences are `duration`s; `duration_cast` converts
to ms/us/ns. A small `benchmark` helper returns the result so the compiler
cannot optimise the work away.
**Expected output:** ten dice rolls, reals, normals and coin flips fixed by
seed 42, a shuffle repeated with the same seed, and two timed loops.

### 09_regex.cpp — pattern matching
**Headers used:** `<regex>`, `<string>`.
**Key points:** `regex_match` needs the whole string, `regex_search` finds a
pattern anywhere; parentheses create capture groups read from `std::smatch`
(`m[1]`, `m[2]`); `sregex_iterator` walks every match; `regex_replace` rewrites
with `$1`. Build each `std::regex` once and reuse it. Two mini parsers show
the pattern in practice: `key=value` lines and `dd/mm/yyyy` → ISO dates.
**Expected output:** match/search answers, extracted year/month/day, found
emails and words, two replacements, the parsed key/value table and dates.

---

## Key takeaways
- Pick the container that matches your access pattern.
- Iterators are the one lingua franca for all containers.
- `sort`, `count_if`, `accumulate`, `find`, `transform` cover most work.
- Seed `mt19937` once and log the seed; time with `steady_clock`.
- Regex: `match` = whole string, `search` = anywhere, groups via `smatch`.
- Pick the container by access pattern: `vector` by default, `map`/`set` for sorted keys.
- Iterators are the common language between containers and `<algorithm>`.
- Seeded `<random>` engines make experiments reproducible; `<chrono>` measures them.

## Next module

**08_templates_generics** - function and class templates, specialization, variadics.
