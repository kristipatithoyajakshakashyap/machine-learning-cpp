# 11_capstone — The Graduation Project

## Purpose
Everything so far meets here in a small data pipeline: statistics, CSV
loading, normalization, logging, and tests — the bridge between "C++ basics"
and the data-science material that comes next in this course.

## Prerequisites

Every earlier module: strings, vectors, classes, exceptions, files, logging, templates.

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_stats_calculator.cpp` | `conc_stats_calculator` | Statistics you can trust: the numbers behind every data table | prints only |
| `02_csv_dataset_reader.cpp` | `conc_csv_dataset_reader` | Dataset loading: text file -> rows of numbers in memory | `dataset.csv` (written, then read back) under `results/02_csv_dataset_reader_results/` |
| `03_dataset_normalizer.cpp` | `conc_dataset_normalizer` | Normalization: rescaling a column so every value lands in [0, 1] | prints only |
| `04_logger_integration.cpp` | `conc_logger_integration` | Putting it together: a tiny end-to-end pipeline with a logger | `capstone.log` under `results/04_logger_integration_results/` |
| `05_projects_tests.cpp` | `conc_projects_tests` | Your own test harness: PASS/FAIL + a summary, no framework needed | prints only |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/11_capstone/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_projects_tests
.\build\01_get_to_know\11_capstone\conc_projects_tests.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** mean, median, min/max, standard deviation.
- **File 02:** CSV -> `struct Sample` dataset.
- **File 03:** min-max normalization of a column.
- **File 04:** a mini pipeline narrated by the module-06 logger.
- **File 05:** a tiny self-built test harness (PASS/FAIL + summary).

---

## Files in this folder

### 01_stats_calculator.cpp
**Headers used:** `<algorithm>`, `<cmath>`, `<iomanip>`, `<vector>`.
**Functions:** `mean` (sum / count), `median` (middle of a *sorted copy*;
averages the two middles for even counts), `population_stddev`
(root of mean squared deviation).
**Key points:** the median is the same 2 4 4 4 5 5 7 9 sample sorted and
unsorted here; `setprecision(3)` keeps the numbers short.
**Expected output:** sample/sorted rows, `mean = 5`, `median = 4.5`,
`min = 2`, `max = 9`, `std dev = 2`.

### 02_csv_dataset_reader.cpp
**Headers used:** `<fstream>`, `<iomanip>`, `<string>`, `<vector>`.
**Types:** `struct Sample { double height; double weight; }` and
`struct Dataset { std::vector<Sample> rows; }`.
**Functions:** `split` (reused from 05), `load_csv(path)` — skips the
header, parses with `std::stod`, returns a Dataset (empty on failure).
**Key points:** the file is written first so the lesson is self-contained;
all work happens in memory afterwards.
**Expected output:** file written, `loaded 4 samples`, one row per record,
`average height = 171.25`, `tallest sample has height 182.00`.

### 03_dataset_normalizer.cpp
**Headers used:** `<algorithm>`, `<iomanip>`, `<vector>`.
**Functions:** `min_of`/`max_of`, `min_max_normalize` (value−min)/(max−min).
**Key points:** normalizing per-column puts every column in [0,1] so
columns with different units can be compared; the original vector is kept
untouched (a copy is normalized).
**Expected output:** the heights with min/max, the normalized row
`0.5 1 0 0.708`, and a note that min maps to 0 and max to 1.

### 04_logger_integration.cpp
**Headers used:** `<cstdio>`, `<ctime>`, `<filesystem>`, `<fstream>`,
`<optional>`, `<string>`, `<vector>`.
**Types:** `Logger` (console + file, INFO/WARN/ERROR, fixed clock for
repeatable output).
**Functions:** `column_mean(vector<double>) -> std::optional<double>` —
an empty column reports "no data" through `nullopt` instead of an
exception.
**Key points:** step-by-step narration of read -> compute -> warn -> error
and the optional-based failure path.
**Expected output:** five timestamped pipeline lines ending in
`finished (log in capstone.log)`.

### 05_projects_tests.cpp
**Headers used:** `<algorithm>`, `<cmath>`, `<vector>`.
**Pattern:** a `CHECK(condition, message)` macro counts every check; the
logic under test (mean/median/normalize) is copied from 01 and 03; doubles
are compared with `1e-9` tolerance.
**Key points:** `return 0` only when nothing failed — the executable's
exit code is itself testable (used by the final gate).
**Expected output:** five `[PASS]` lines and `5 checks, 0 failed`.

---

## Key takeaways
- Statistics + CSV + normalize = the skeleton of every data table you will
  meet later.
- Logging keeps pipelines explainable; optional keeps failures honest.
- A 30-line test harness that exits non-zero on failure beats no tests forever.

## Next module

**12_debugging_and_testing** - contracts, sanitizers, a debugger recipe and a small test framework; then **13_numerical_computing**.
