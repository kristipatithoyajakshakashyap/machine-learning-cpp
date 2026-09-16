# Machine learning with C++17

This course implements models, preprocessing, validation, plots and persistence in C++. There are no Python scripts or package requirements. CMake builds each numbered lesson as an executable. The `course` preset in the repository-root `CMakePresets.json` configures, builds and tests everything with three commands (see below).

## Learning path

1. `01_supervised/01_regression`: linear/ridge/lasso/polynomial, kNN, trees, forests, gradient boosting, MLP, Elastic Net, linear/RBF SVR and Gaussian processes.
2. `01_supervised/02_classification`: logistic, kNN, SVM, Gaussian NB, trees, forests, AdaBoost, gradient boosting, MLP, evaluation, LDA, QDA and imbalanced classification.
3. `01_supervised/03_model_diagnostics` and `04_ensembles`: permutation importance, learning curves, feature selection, hyperparameter search with nested CV, voting and stacking.
4. `02_unsupervised`: K-means, hierarchical clustering including Ward, silhouette/stability evaluation, DBSCAN, Gaussian mixtures, PCA and t-SNE.
5. `03_anomaly_detection`: Isolation Forest, Local Outlier Factor and One-class SVM.
6. `04_time_series`: forecasting baselines, lag features, rolling-origin evaluation, Holt-Winters and ARIMA.
7. `05_text_classification`: tokenization, count/TF-IDF, Multinomial NB and linear classification.
8. `06_recommender_systems`: popularity, item-based collaborative filtering, biased matrix factorization and ranking evaluation.

Each algorithm folder contains theory, worked mathematical intuition, executable demonstrations, implementation guidance, exercises and an end-to-end project. Model behavior shared by several algorithms is compiled once: tree/ensemble internals are in `helper/models`, and kNN/MLP are owned by their regression modules with classification forwarding interfaces. Everything else that is shared (matrices, metrics, optimisers, folds, plots, the archive format, the `Artifacts` output sandbox and the supervised runner) lives in `helper/` and is documented in `helper/README.md`.

## How a module is organised

Every algorithm folder (for example `05_text_classification/03_multinomial_naive_bayes`) follows one layout so you always know where to look:

| Item | What it is |
|---|---|
| `README.md` | Method, core idea, dataset, prerequisites, file table, build/run, results layout, tests, takeaways, next module. |
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | Reading material, in that order. |
| `Model.hpp` (or `<Name>.hpp/.cpp`) | The model itself: `fit`, `predict` (or `probabilities` / `rank_score`), `save`, `load`. Headers used by several modules (`Text.hpp`, `Workflow.hpp`) live in the first module that needs them or in the track folder. |
| `01_theory.cpp`, `02_math_intuition.cpp` | Lessons that only print. No results folder. |
| `03_implementation.cpp` (and any further numbered lessons, e.g. `04_ridge_regression.cpp`) | A worked example on the module's dataset that writes CSV and SVG artifacts. |
| `0N_end_to_end.cpp` | The final project: calls the track's shared workflow (`helper/pipeline/supervised.hpp`, `05_text_classification/Workflow.hpp` or `06_recommender_systems/04_recommender_evaluation/Workflow.hpp`), which splits, selects, evaluates, persists and verifies a reload. Accepts `--quick`. |
| `predict.cpp` | Inference CLI that reloads a saved run (`--predict rows.csv --model <run>/model`, `--predict messages.txt --model <run>/model`, or `--model <run>/model/model.txt --user <id>` for recommenders). |
| `tests/model_test.cpp` | Numerical fixture registered with CTest as `<prefix>_numerical`. The `--quick` project is registered as `<prefix>_workflow`. |
| `CMakeLists.txt` | One `add_lesson_executable(<prefix>_<lesson> <file>.cpp)` per source. The macro links `ml_core` and defines `COURSE_ROOT`, `DATA_DIR` and `RUN_OUTPUT_DIR` for that target. |

Each `.cpp` starts with a header comment stating its purpose, inputs, outputs and CMake target, so you read a source file on its own.

## Build and test

From the repository root, with `D:\msys64\ucrt64\bin` on PATH (the MinGW runtime directory must be on PATH to run the executables):

```powershell
$env:PATH = 'D:\msys64\ucrt64\bin;' + $env:PATH
cmake --preset course          # configure into build/ (MinGW Makefiles, Release, tests on)
cmake --build --preset course  # build every lesson, predict tool and test
ctest --preset course          # run every numerical fixture and --quick workflow
```

`cmake --build --preset course --target <name>` builds one target, and `ctest --preset course -R "text_|rec_"` runs a subset. The equivalent manual commands from this folder are `cmake -S . -B build -G 'MinGW Makefiles' -DCMAKE_CXX_COMPILER=D:/msys64/ucrt64/bin/g++.exe -DCMAKE_BUILD_TYPE=Release`, `cmake --build build --parallel 4` and `ctest --test-dir build --output-on-failure`.

## End-to-end projects

```powershell
build/01_supervised/01_regression/01_linear_regression/lin_end_to_end.exe --quick
build/01_supervised/01_regression/01_linear_regression/lin_end_to_end.exe
build/01_supervised/01_regression/01_linear_regression/lin_predict.exe --predict new_rows.csv --model 01_supervised/01_regression/01_linear_regression/results/08_end_to_end_results/full/model
```

The generic supervised runner reserves a seeded 20% holdout, produces training-only EDA and fits preprocessing independently in every validation fold. Five-fold training CV selects parameters. The winner is refitted before evaluating the holdout once. Classification splits are stratified. Regression targets are standardized inside training and predictions restored to original units. Quick verification uses at most 400 deterministic observations and is labeled separately from full-data runs.

Clustering uses internal selection and resampling stability, not a mislabeled supervised CV loop. Hierarchical clustering and DBSCAN preserve fitted state without inventing native prediction for new observations. Text fits vocabulary and IDF within training folds. Forecasting and recommendation use chronological validation. See their READMEs for metrics and protocols.

## Results and persistence

Every C++ lesson owns `results/<cpp_filename_stem>_results/` inside its algorithm folder. The path is compiled in as `RUN_OUTPUT_DIR`, so the working directory never matters. For example, `03_implementation.cpp` writes to `results/03_implementation_results/`, and `08_end_to_end.cpp` writes to `results/08_end_to_end_results/`. Supervised, text and recommender end-to-end projects keep `full/` (default) and `quick/` (`--quick`) inside that named folder, each with `data/`, `eda/`, `validation/`, `evaluation/`, `model/`, `inference/`, `run_manifest.json`, `report.md` and `execution.log`. Unsupervised and anomaly workflows write directly inside their named end-to-end folder. `predict.cpp` writes to `results/predict_results/`. Lessons that only print (theory, math intuition) have no results folder, and no lesson writes an `output.txt`: the console shows progress, `report.md` holds the interpretation. Repeating a named run replaces its artifacts. Copy a run inside the algorithm folder before comparing manual experiments.

Saved evidence includes data quality, distributions, feature relationships, fold/candidate scores, predictions or assignments, evaluation metrics, plots, configuration, model state, learned preprocessing and reload checks. Read the generated report for interpretation and limitations. Numeric inference reads named feature columns from the saved schema. Text and recommender CLIs document their inputs.

Archives are versioned text with 17 significant digits for numeric round trips. They are course formats, not binary memory dumps or interchangeable formats from another ML library.

## Verification and historical outputs

CTest executes C++ mathematical fixtures, invalid-input checks, leakage tests, serialization checks and complete workflow tests. A passing build alone does not establish numerical correctness.

Every results folder contains only data, plots and model files produced by the current code. Lessons that only print (theory, math intuition) have no results folder. Historical `previous_outputs/` folders were removed. Expected-output source comments are historical teaching snapshots. Corrected optimization, initialization, preprocessing and splitting change these numbers. They are not correctness goldens. Current runs in each `results/` folder are the executed evidence.