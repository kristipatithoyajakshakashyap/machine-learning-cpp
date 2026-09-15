# 05_file_handling — Saving and Loading Data

## Purpose
Programs that forget their data are useless. This module teaches the three
ways C++ talks to disk: **text files**, **binary files**, and **CSV tables**,
plus the `std::filesystem` toolbox for paths and folders.

All examples write into `results/<source stem>_results/` (the `RUN_OUTPUT_DIR` path injected
by CMake) so the source folder stays clean. Re-running a lesson rewrites its
files — no cleanup needed.

## Prerequisites

Modules 01-04 (strings, vectors, classes, exceptions).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_text_files.cpp` | `conc_text_files` | Text files: writing, reading, appending | `notes.txt` under `results/01_text_files_results/` |
| `02_binary_files.cpp` | `conc_binary_files` | Binary files: saving and loading raw bytes (exact copies, very fast) | `records.bin` under `results/02_binary_files_results/` |
| `03_csv_files.cpp` | `conc_csv_files` | CSV files: the most common plain-text table format | `students.csv` under `results/03_csv_files_results/` |
| `04_filesystem.cpp` | `conc_filesystem` | std::filesystem: working with paths, folders, and file properties | a temporary `demo_dir/` tree (created, inspected, then removed) under `results/04_filesystem_results/` |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/05_file_handling/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_filesystem
.\build\01_get_to_know\05_file_handling\conc_filesystem.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- **File 01:** text files — write, read, append.
- **File 02:** binary files — exact raw bytes.
- **File 03:** CSV — reading a table one row at a time (the pattern you will
  use to load data into a program later).
- **File 04:** `std::filesystem` — folders, sizes, rename, delete.

---

## Files in this folder

### 01_text_files.cpp — writing and reading
**Headers used:** `<fstream>` (`ifstream`/`ofstream`), `<iostream>`,
`<string>`.
**Functions/key points:** `ofstream` (overwrite mode) writes lines;
`if (!out)` checks the file really opened; `std::getline` reads a whole line;
`ofstream(..., std::ios::app)` appends; `.close()` flushes and releases.
**Expected output:** "wrote 3 lines", the three lines read back, "read 3
lines back", and "after appending one line, the file has 4 lines".

### 02_binary_files.cpp — exact bytes
**Headers used:** `<cstddef>`, `<cstring>` (`strncpy`), `<fstream>`,
`<iomanip>` (`setprecision`), `<iostream>`, `<string>`.
**Types/functions:** a fixed-layout `struct Record { int, double, char[16] }`;
`out.write(ptr, size)` and `in.read(ptr, size)` move whole structs as raw
bytes; `reinterpret_cast` tells `write()`/`read()` "here are the bytes".
**Key points:** binary files reproduce memory exactly and are fast; the
struct's layout must never change between save and load.
**Expected output:** "wrote 3 binary records, each 32 bytes", the three
records read back (`1.50`, `2.25`, `-7.00`), and a confirmation the
round-trip was byte-for-byte identical.

### 03_csv_files.cpp — the CSV table
**Headers used:** `<fstream>`, `<iomanip>`, `<iostream>`, `<string>`,
`<vector>`.
**Function:** `split(line)` breaks `"Ada,101,88"` into columns using commas.
**Key points:** a CSV row is one line and commas separate columns; skip the
header before counting data; convert numeric text with `std::stoi`;
accumulate columns to compute a summary (here an average score).
**Expected output:** the header written, three padded rows (`Ada / Grace /
Linus` with id and score), "parsed 3 rows", and `average score = 87.667`.

### 04_filesystem.cpp — paths and folders
**Headers used:** `<algorithm>` (`sort`), `<filesystem>`, `<fstream>`,
`<iostream>`, `<string>`, `<vector>`.
**Key points:** alias `namespace fs = std::filesystem`; `create_directories`
builds folders including parents; `directory_iterator` lists a folder (results
sorted here for a stable screen order); `.filename()` gives just the tail of a
path — use `.string()` to print it without quotes; `file_size`, `rename`,
`remove` finish the chores.
**Expected output:** existence/directory checks, the sorted folder listing,
`a.txt has size 6 bytes`, rename then removal both confirmed.

---

## Key takeaways
- Always check a file `if (!stream)` before trusting it.
- `read`/`write` for binary, `<<`/`getline` for text, `split`+`stoi` for CSV.
- Print paths with `.string()` — the raw `operator<<` adds quotes on Windows.
- `std::ofstream`/`std::ifstream` handle text; open in binary mode for exact bytes.
- CSV parsing is `getline` per row, then split on commas and convert each field.
- `std::filesystem` creates, inspects, renames and removes paths portably.

## Next module

**06_logging** - log levels, timestamps, file logging and a reusable `Logger` class.
