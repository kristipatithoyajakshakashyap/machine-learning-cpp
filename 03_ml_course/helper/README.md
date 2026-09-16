# Course helpers (`ml_core`)

Everything under `helper/` is the course's stand-in for numpy, pandas,
matplotlib and the non-model parts of scikit-learn. All `.cpp` files here are
compiled once into the static library `ml_core` (see the `add_library` call in
`../CMakeLists.txt`). Every lesson executable links it and includes headers
with paths relative to `03_ml_course/`, for example
`#include "helper/reporting/artifacts.hpp"`. `helper/models/tree_models.cpp`
is the one exception: it is its own library, `ml_tree_models`, linked only by
the tree and ensemble modules.

Nothing in this folder writes output on its own initiative. Files are written
only through `ml::Artifacts`, and only into the results folder of the lesson
that called it.

## What each subfolder provides

| Folder | Files | Provides |
|---|---|---|
| `data/` | `datasets.hpp/.cpp` + checked-in CSVs (`wine`, `breast_cancer`, `iris`, `penguins`, `titanic`, `diabetes`, `california_housing`, `tips`, `air_passengers`, `flights`) | `ml::Dataset` (dense `X`, `y`, feature names) and one loader per CSV so every lesson sees the same column order. |
| `dataframe/` | `dataframe.hpp/.cpp` | `ml::DataFrame`, a small column store with `read_csv`, `numeric()` and `strings()`. The loaders in `data/` are built on it. |
| `math/` | `matrix.hpp/.cpp`, `metrics.hpp/.cpp`, `optim.hpp/.cpp` | Dense `ml::Mat` / `ml::Vec` algebra with dimension checks. Classification and regression metrics. Deterministic gradient descent, SGD and the shared seeded RNG (`seed_rng`, `shuffle_indices`). |
| `eval/` | `cross_validation.hpp/.cpp`, `diagnostics.hpp`, `feature_selection.hpp`, `search.hpp` | Seeded k-fold / stratified folds. Permutation importance and learning curves. Filter, wrapper and L1 feature selection. Grid, random and nested hyper-parameter search. |
| `models/` | `tree_models.hpp/.cpp` | CART decision tree plus the random forest, AdaBoost and gradient boosting learners shared by the tree modules (`ml_tree_models`). |
| `persistence/` | `archive.hpp` | The single text archive format used by every `save()` / `load()` in the course. |
| `pipeline/` | `supervised.hpp` | `run_supervised<Factory>()`, the shared end-to-end project and `--predict` CLI for every dense supervised module. |
| `preprocessing/` | `preprocessor.hpp` | `ml::Preprocessor`: median imputation followed by z-scoring, fitted on training rows only. |
| `plot/` | `plot_svg.hpp/.cpp` | `ml::Plot`: line, scatter, bar and histogram charts rendered to self-contained SVG text. |
| `reporting/` | `artifacts.hpp/.cpp` | `ml::Artifacts`: the sandboxed results directory, `write()` / `figure()`, standard EDA outputs and data fingerprints. `csv_quote` / `json_quote`. |

## Contracts other code relies on

Each header states its own contract in its top comment. The load-bearing
ones are summarised here so you write a new module against them.

### Dense supervised model (`pipeline/supervised.hpp`)

`run_supervised(dataset, RUN_OUTPUT_DIR, name, candidates, make,
classification, argc, argv)` takes a factory `make(double) -> Model`. The
model must provide

```cpp
void fit(const ml::Mat &X, const ml::Vec &y);
ml::Vec predict(const ml::Mat &X) const;
void save(std::ostream &) const;
void load(std::istream &);
// optional: ml::Mat predict_proba(const ml::Mat &) const;  // ROC, log loss, PR, calibration
// optional: ml::Vec decision_function(const ml::Mat &) const; // ROC AUC only
```

The runner reserves a seeded 20% holdout, runs EDA on training rows only,
refits `Preprocessor` and `TargetTransform` inside every CV fold, refits the
winning candidate on all training rows, scores the holdout once with
bootstrap intervals, then archives `model/model_state.txt`,
`preprocessing_state.txt`, `target_state.txt` and `features.txt` and verifies
that reloading them reproduces the holdout predictions. `--quick` uses at
most 400 rows and writes to `quick/` instead of `full/`. `--predict
rows.csv --model <run>/model` reads named feature columns and writes
`predictions.csv`.

The text and recommender tracks use the same shape of contract but their own
runner: see `05_text_classification/Workflow.hpp` and
`06_recommender_systems/04_recommender_evaluation/Workflow.hpp`, and the
contracts written in `05_text_classification/README.md` and
`06_recommender_systems/01_popularity_baseline/Model.hpp`.

### Archive format (`persistence/archive.hpp`)

- `archive::write(out, a, b, ...)` / `archive::read(in, a, b, ...)`
  serialise arguments in order. Arithmetic types, enums, `std::string`,
  `std::vector<T>` and `std::pair<A, B>` are handled recursively. Any other
  type must provide `save(std::ostream&) const` and `load(std::istream&)`.
- Doubles use 17 significant digits, so a round trip is bit-exact. The
  workflows assert this.
- `save_file` / `load_file` add the magic line `MLCPP_ARCHIVE_V1`. Models add
  their own type tag as the first value and reject foreign archives.
- Vectors are length-prefixed and lengths above 10,000,000 are rejected.
  Every stream failure throws `std::runtime_error`.

### Preprocessor (`preprocessing/preprocessor.hpp`)

`fit()` must only ever see training rows. `transform()` is then applied to
validation, test and inference rows so no statistics leak. The fitted state
is archived with the model so `--predict` reproduces the training pipeline.

### Artifacts path rules (`reporting/artifacts.hpp`)

- Construct with `Artifacts(RUN_OUTPUT_DIR, name)` where `name` is `"full"`,
  `"quick"` or `"."` (write directly into the root). `RUN_OUTPUT_DIR` is a
  compile definition set per target and equals
  `<module>/results/<cpp_filename_stem>_results`, so the working directory
  never matters.
- The run name and every relative path must be non-empty, relative and free
  of `..`. Anything else throws `std::invalid_argument`, so a lesson never
  writes outside its own results folder.
- `path()` creates parent directories, so nested names such as
  `evaluation/figures/roc.svg` need no preparation.
- `write()` is binary and flushes. A failed write throws rather than leaving
  a partial file. `figure()` renders a `Plot` and stores it through `write()`.
- `eda()` writes `data/schema.json`, `<prefix>/findings.md`,
  `<prefix>/tables/*.csv` and `<prefix>/figures/*.svg`. `fingerprint()`
  hashes the dataset for `run_manifest.json`.

## Determinism

Every stochastic routine (fold shuffling, SGD, forests, permutation
importance, random search) draws from the shared seeded RNG in
`math/optim.hpp` or a local `std::mt19937` seeded by the caller, and the
shared RNG is re-seeded before every fit. The same build therefore reproduces
the same results files run after run, which is what the `*_workflow` CTest
entries depend on.

## Tests

`../tests/numerical/` exercises the helpers directly. `core.cpp`
(`ml_correctness_tests`, CTest `ml_correctness`) covers splits, the
preprocessor, metrics, the archive and the dense supervised models with
hand-computable fixtures. `compare_predictions.cpp` (`ml_compare_predictions`)
backs the `reload_<module>` CTest entries by diffing the workflow's in-memory
predictions against the predict tool's reloaded ones. The `*_guards.cpp`
files (`ml_knn_guards`, `ml_nb_guards`, `ml_svm_guards`) feed invalid input
and corrupt archives to kNN, Naive Bayes and SVM models and expect
exceptions. Run them with `ctest --preset course -R "ml_"`.