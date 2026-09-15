# File guide: what you write, what is generated, and when

## 1. Files you write by hand

### Repository level (written once, rarely touched)

| File | Purpose | When to edit |
|---|---|---|
| `CMakeLists.txt` (root) | adds the three tracks, writes `build/lesson_manifest.txt` | only when adding a new track |
| `CMakePresets.json` | `course` preset: MinGW Makefiles, Release, tests on | only when changing compiler/generator |
| `README.md` | entry point, build commands | when the workflow changes |
| `docs/project_structure.md`, `docs/learning_path.md`, `docs/file_guide.md` | the only three docs | when modules are added or the order changes |
| `scripts/run_all_lessons.ps1` | runs every lesson in manifest order with the right arguments | when a lesson needs new arguments or stdin |
| `scripts/verify_artifacts.ps1` | checks every results file is non-empty and ML full runs are complete | when required ML artifacts change |
| `.vscode/settings.json`, `.vscode/tasks.json` | editor CMake root, single-file g++ task | optional |

### Track level

| File | Purpose |
|---|---|
| `01_get_to_know/CMakeLists.txt`, `02_data_science/CMakeLists.txt`, `03_ml_course/CMakeLists.txt` | define `add_lesson_executable(target source.cpp)`, compile flags, shared libraries, `DATA_DIR` |
| `01_get_to_know/README.md`, `02_data_science/README.md`, `03_ml_course/README.md` | track overview |
| `02_data_science/include/dsts/*.hpp`, `02_data_science/src/*.cpp` | the `dsts` toolkit (Series, DataFrame, CSV, stats, cleaning, EDA, plots, join, features, sampling, time series, PCA) |
| `03_ml_course/helper/**` | shared ML library: dataframe, math, datasets, eval, reporting, plot, persistence archive, preprocessing, supervised pipeline |
| `02_data_science/data/*.csv`, `03_ml_course/helper/data/*.csv`, `03_ml_course/05_text_classification/data/*`, `03_ml_course/06_recommender_systems/data/*` | checked-in datasets |
| `03_ml_course/05_text_classification/data/inference_messages.txt` | input lines for text `predict` targets |
| `02_data_science/tests/regression.cpp`, `03_ml_course/tests/**`, `03_ml_course/02_unsupervised/tests/**` | CTest fixtures |

### Module level: C++ basics and data science (`01_get_to_know/NN_*`, `02_data_science/NN_*`)

| File | Required | Content |
|---|---|---|
| `README.md` | yes | what the module teaches, list of lessons, targets |
| `NN_topic.cpp` | yes, one per lesson | self-contained `main()`; header comment states WHAT YOU LEARN and EXPECTED OUTPUT |
| `CMakeLists.txt` | yes | one `add_lesson_executable(<target> NN_topic.cpp)` per lesson |
| shared `.hpp` (e.g. `11_final_pipeline/pipeline.hpp`, `12_advanced_eda/eda_workflow.hpp`) | only when several lessons share one implementation | the real code; lesson files become 3-line `main()` wrappers |
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | only for capstone modules (`11_final_pipeline`) | same meaning as in ML modules |
| `predict.cpp` | only when a model is saved (`11_final_pipeline`) | reload saved state in a fresh process, reproduce predictions |

### Module level: ML algorithm (`03_ml_course/<group>/<NN_algorithm>/`)

Every algorithm folder has this fixed set. Create them in this order.

| Order | File | Content |
|---|---|---|
| 1 | `README.md` | one paragraph: problem type, dataset, what the module shows, targets |
| 2 | `theory.md` | assumptions, strengths, failure modes, when to prefer another model |
| 3 | `math_intuition.md` | objective, derivation, update/closed form, complexity, tiny hand example |
| 4 | `<Algorithm>.hpp` + `<Algorithm>.cpp` | the model class: `fit`, `predict`, `save`/`load` via `helper/persistence/archive.hpp`. Some modules use a header-only `Model.hpp` or a shared `Workflow.hpp` instead |
| 5 | `implementation.md` | walk through the class: data structures, numerical choices, edge cases |
| 6 | `01_theory.cpp` | prints the theory summary and demonstrates assumptions on tiny data |
| 7 | `02_math_intuition.cpp` | reproduces the hand example numerically |
| 8 | `03_implementation.cpp` | fits the class on a small dataset and prints/saves a diagnostic (often an SVG) |
| 9 | optional `04..07_<variant>.cpp` | extra lessons (ridge/lasso/polynomial, RBF kernel, depth curve, n_estimators curve, multiclass) |
| 10 | `NN_end_to_end.cpp` | one call into the shared pipeline: load data, hold-out split, EDA, CV model selection, final fit, evaluation, save model, reload check. `NN` is the next free number (04, 05, 06 or 08) |
| 11 | `predict.cpp` | fresh-process inference from the saved model; supervised modules `#include` the end-to-end file and gate on `--predict`/`--model` |
| 12 | `tests/model_test.cpp` | numerical fixture with hand-computed expectations (required for every module that ships a model class) |
| 13 | `exercises.md` | tasks that change data or parameters and ask for an explanation |
| 14 | `CMakeLists.txt` | `set(ML_MODULE_LIB ...)`, `add_library(ml_<x> STATIC <Algorithm>.cpp)`, one `add_lesson_executable` per `.cpp`, `add_test` for the numerical fixture and the `--quick` workflow |

Group-level files (`01_supervised/CMakeLists.txt`, `02_unsupervised/CMakeLists.txt`, ...) only `add_subdirectory` each module and are edited when a module is added.

## 2. Files that are generated (never edit, never commit by hand)

| Path | Generated by | Contents |
|---|---|---|
| `build/**` | `cmake --preset course` + `cmake --build --preset course` | Makefiles, object files, `.exe` for every target, CTest metadata |
| `build/lesson_manifest.txt` | CMake configure (root `CMakeLists.txt`) | one line per lesson: `target|exe_path|results_dir` |
| `<module>/results/<lesson>_results/` | running the lesson (directly or via `scripts/run_all_lessons.ps1`) | only files the lesson writes itself (CSV, SVG, JSON, MD, model state); folder name is the source file stem plus `_results` (`04_csv_io.cpp` -> `04_csv_io_results/`). Print-only lessons have no folder |
| `results/NN_end_to_end_results/full/` and `quick/` | `NN_end_to_end` run without and with `--quick` | `data/` (split, schema, hold-out features), `eda/` (findings, tables, figures), `validation/` (candidate scores, selected parameters), `model/` (saved state), `evaluation/` (metrics.json, predictions.csv, per-class, error analysis), `inference/reload_verification.json`, `report.md`, `run_manifest.json`, `execution.log` |
| `results/predict_results/` | `predict` targets | `predictions.csv` (or `next_forecast.csv`, `recommendations.csv`) |
| `results/verify_reload_results/` | CTest (`unsupervised_reload`) | reload verification for unsupervised and anomaly models |
| `results/regression_results/` | `dsts_regression_tests` | CSV round-trip and pipeline test outputs |
| `02_data_science/11_final_pipeline/results/05_eval_model_results/` | `s11_05_eval` | Titanic features, PCA scores, metrics, `model_state.txt` used by `s11_predict` |
| SVG / CSV / JSON / MD inside any `results/` | the owning lesson | plots, tables, reports |

Delete any of these and rerun; nothing else depends on them being present except the `predict` and reload targets, which need their module's end-to-end run first.

## 3. When to create which file

| Situation | Create | Then |
|---|---|---|
| New C++ basics or data science lesson | `NN_topic.cpp` in the module | add `add_lesson_executable` line to module `CMakeLists.txt`; list it in module `README.md`; reconfigure; run it |
| New data science module | folder `NN_name/` with `README.md`, `CMakeLists.txt`, lessons | `add_subdirectory` in `02_data_science/CMakeLists.txt`; add row to `docs/learning_path.md` and tree in `docs/project_structure.md` |
| New ML algorithm | folder with the 14 files in section 1 | `add_subdirectory` in the group `CMakeLists.txt`; add to the algorithm order in `docs/learning_path.md`; add to tree in `docs/project_structure.md`; run `scripts/run_all_lessons.ps1 -Filter <prefix>` then `ctest --preset course` and `scripts/verify_artifacts.ps1 -RequireFullML` |
| Lesson must write files | use `RUN_OUTPUT_DIR` (absolute, injected by CMake) and call `std::filesystem::create_directories(RUN_OUTPUT_DIR)` first; ML code uses `ml::Artifacts(RUN_OUTPUT_DIR, ...)` which creates it | never hardcode paths; never write relative to the working directory |
| Lesson needs stdin or arguments | add a case in `scripts/run_all_lessons.ps1` | keep the lesson runnable by hand with the same arguments |
| New dataset | drop the file in the track's `data/` folder | reference it via `DATA_DIR` (data science and ML) or `COURSE_ROOT` (text, recommender) |
| New shared helper | `.hpp`/`.cpp` in `02_data_science/include+src` or `03_ml_course/helper/<area>/` | add the `.cpp` to the `dsts` or `ml_core` library list in the track `CMakeLists.txt` |
| Model class changes | update `tests/model_test.cpp` expectations first | rerun the end-to-end (full and quick) so saved state, `report.md` and `predict_results` match the code |

## 4. Standard regeneration sequence

```
cmake --preset course
cmake --build --preset course
./scripts/run_all_lessons.ps1                 # all lessons; ML end-to-end full + quick; predicts last
ctest --preset course                         # all numerical, integration and reload tests
./scripts/verify_artifacts.ps1 -RequireFullML # every results file non-empty, all ML full runs present
```

After this sequence there are no empty folders anywhere in the tree, every results folder is named `<source stem>_results`, and every file under `results/` was written by a lesson executable.
