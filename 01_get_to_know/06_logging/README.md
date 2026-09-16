# 06_logging: Recording What Your Program Does

## Purpose
Logging is how a running program talks back: it writes a trail of
"what happened, when, and how bad" that survives long after the console
window closes. This module builds up from tiny print helpers to a
reusable `Logger` class before the later modules put it to work.

## Prerequisites

Module 05_file_handling (writing text files) and 03_oop (classes).

## Lessons

| File | Target | What it teaches | What it writes |
|------|--------|-----------------|----------------|
| `01_logging_basics.cpp` | `conc_logging_basics` | Logging basics: labeled messages with severity levels | prints only |
| `02_logging_to_file.cpp` | `conc_logging_to_file` | Logging to a file: a permanent record of what the program did | `run.log` under `results/02_logging_to_file_results/` |
| `03_multichannel_logger.cpp` | `conc_multichannel_logger` | One Logger class that writes to BOTH console and file, with a minimum-level filter | `runner.log` under `results/03_multichannel_logger_results/` |
| `04_reusable_logger.cpp` | `conc_reusable_logger` | A production-flavored Logger: timestamps, plain-text helper methods, console + file logging | `event.log` under `results/04_reusable_logger_results/` |

## Build and run

Configure once from the repository root, then build the lesson you want and
run its executable from `build/01_get_to_know/06_logging/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target conc_reusable_logger
.\build\01_get_to_know\06_logging\conc_reusable_logger.exe
```

`cmake --build --preset course` with no `--target` builds every lesson in the
course. Lessons that write files create `results/<source stem>_results/` next
to their source on first run (the path is injected as `RUN_OUTPUT_DIR`).

## Suggested pace
- File 01: the three severity levels `INFO` / `WARN` / `ERROR`.
- File 02: text to a file instead of a screen + a timestamp per line.
- File 03: one `Logger` class writing to console and file with a
  minimum-level filter.
- File 04: the finished reusable `Logger`. Timestamps, helper methods,
  runtime filter.

---

## Files in this folder

### 01_logging_basics.cpp: levels and helpers
Headers used: `<iostream>`, `<string>`.
Functions: `log_info`, `log_warn`, `log_error`. One tiny helper per level so every message line is formatted the same way (`[INFO ]` style, `ERROR` tag aligned to 5 chars).
Key points: INFO = normal progress, WARN = unusual but survivable, ERROR = something failed.
Expected output: five labeled lines describing a tiny scoring job, ending with `[INFO ] program finished`.

### 02_logging_to_file.cpp: the file gets the details
Headers used: `<cstdio>` (`snprintf`), `<ctime>` (`std::tm`), `<filesystem>`, `<fstream>`, `<iostream>`, `<string>`.
Functions: `format_time(const std::tm&)` renders a clock value as `2026-09-10 14:30:05`. `append_log(text)` opens `run.log` with `std::ios::app` (appending, so old lines survive every run) and adds a timestamped line. `count_lines(path)` returns how many lines a file has.
Key points: the timestamp is built from a fixed `std::tm` so the demo output is identical everywhere. A real program passes `std::time(nullptr)`. The file is deleted at the start of `main` so re-runs stay reproducible.
Expected output: 3 lines written, 2 appended on top, `final file line count = 5`, and the tail of the log printed (`last log line is finished`).

### 03_multichannel_logger.cpp: console + file, one class
Headers used: `<filesystem>`, `<fstream>`, `<iostream>`, `<string>`.
Type/element: `enum Level { kInfo, kWarn, kError }` and the `Log` class with `write(level, msg)`.
Key points: `write` returns early when `lvl < min_`, so quiet levels are skipped. Every accepted message is appended to `runner.log` and echoed to `std::cout`. The tag function turns a level into `"INFO "`/`"WARN "`/`"ERROR"` (pad with a space so columns line up).
Expected output: only WARN and ERROR lines appear because the minimum level is `kWarn`. The INFO line is suppressed and the log file matches.

### 04_reusable_logger.cpp: the finished Logger
Headers used: `<cstdio>`, `<ctime>`, `<filesystem>`, `<fstream>`, `<iostream>`, `<string>`.
Class `Logger`: `info()` / `warn()` / `error()` helper methods, `set_min_level(Level)` to change the filter at runtime, `write(...)` stamps each line with the formatted time, appends to `event.log`, and prints to the console.
Key points: call sites read like sentences. The time format function is shared with 02. A real program would feed `std::time(nullptr)`.

---

## Key takeaways
- Three severity levels and keep their print tags the same width.
- Append mode (`std::ios::app`) never deletes history. Delete the file
  yourself when you want a clean slate.
- A level filter (`min_level`) is the cheap way to quiet a program.

- Severity levels let you filter noise without deleting log statements.
- A logger that writes both console and file keeps a permanent record of every run.
- Timestamps and a runtime-adjustable level turn a toy logger into a reusable tool.

## Next module

07_stl_core - containers, iterators, algorithms, `<random>`, `<chrono>` and `<regex>`.