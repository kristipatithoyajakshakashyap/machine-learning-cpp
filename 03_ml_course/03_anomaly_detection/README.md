# Anomaly detection in C++17

## What this track teaches

Anomaly (outlier) detection scores rows by how unusual they are without any labelled examples of
"unusual". Three complementary views are implemented from scratch: Isolation Forest (anomalies
are easy to isolate with random splits), Local Outlier Factor (anomalies sit in regions much
sparser than their neighbours) and One-class SVM (anomalies fall outside a kernel boundary
that encloses the bulk of the data). All three share one experimental protocol with a seeded
60/20/20 split. Preprocessing and the model are fitted on the training partition. A review-budget
threshold is calibrated as the 95th percentile of the validation scores. The untouched test
partition is scored and flagged once.

## Dataset and why

Palmer Penguins (`helper/data/penguins.csv`, 344 rows, four numeric measurements, missing values
kept for the imputer). It carries no anomaly labels, which is the realistic situation: the projects
report score distributions and flag rates, never precision or recall, and the reports say so.
Species labels are not anomaly ground truth and are never used for fitting or calibration.
Implementation lessons use synthetic 2-D clusters with planted outliers so the expected ranking is
known.

## Prerequisites

`../02_unsupervised/` (standardization, the shared `workflow.hpp`, `predict_input.hpp`, the test
binaries). Euclidean distance and basic tree/kernel intuition from `../01_supervised/`.

## Modules

| Path | Target prefix | Method | Score meaning | Predict tool |
|---|---|---|---|---|
| `01_isolation_forest/` | `uif_` | 100 random trees on 256-row subsamples. Short average path = anomalous | in (0, 1), higher = more anomalous | `uif_predict` |
| `02_local_outlier_factor/` | `ulof_` | k = 20 nearest neighbours, local reachability density ratio, novelty mode | LOF around 1 = normal, >> 1 = anomalous | `ulof_predict` |
| `03_one_class_svm/` | `uocsvm_` | RBF kernel boundary with nu = 0.05, gamma = 1/p | rho - decision value, higher = more anomalous | `uocsvm_predict` |

Shared code lives in `../02_unsupervised/`: `workflow.hpp` (`load_penguins`, `manifest`, `eda`,
`snapshot`, `verify`), `validation.hpp` (`validate_dense`, `distance2`), `predict_input.hpp`
(`read_numeric_rows`, `write_predictions`) and `tests/` (`unsupervised_tests`,
`unsupervised_reload`, `penguin_inference.csv`). The CTest entries for this track are registered
in `../02_unsupervised/tests/CMakeLists.txt`.

## Build and run

From the repository root with `D:\msys64\ucrt64\bin` on `PATH`:

```powershell
cmake --preset course
cmake --build --preset course --target uif_end_to_end uif_predict unsupervised_tests unsupervised_reload
build\03_ml_course\03_anomaly_detection\01_isolation_forest\uif_end_to_end.exe
build\03_ml_course\03_anomaly_detection\01_isolation_forest\uif_predict.exe 03_ml_course\02_unsupervised\tests\penguin_inference.csv
```

Replace `uif` by `ulof` or `uocsvm`. The end-to-end programs take no flags (no `--quick`, a run
takes seconds). `<prefix>_predict <numeric_csv>` reads a header-less CSV with the four features in
the order of `model/feature_schema.csv` (raw units, blank/`NA`/`nan` allowed) and writes one score
and one 0/1 flag per row using the saved validation-calibrated threshold.

## Results layout

Every lesson writes only inside its module under `results/<cpp-stem>_results/`. The project run
`results/04_end_to_end_results/` holds `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv` (train/validation/test membership), `eda/` (training rows only),
`validation/{calibration_scores.csv, threshold.json}`, `evaluation/{predictions.csv, metrics.json,
figures/scores.svg}`, `model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`. Predict tools write `results/predict_results/
new_predictions.csv`. The fresh-reload tests write
`results/verify_reload_results/fresh_process_verification.json`.

## Tests

```powershell
ctest --preset course -R unsupervised_numerical   # unit fixtures for all three models
ctest --preset course -R "uif|ulof|uocsvm"         # workflow, fresh reload, new rows
```

## Key takeaways

- Without labels, a threshold is a budget decision: calibrate it on validation scores, never on
  the rows you then report.
- The three methods disagree on what "unusual" means (isolation, local density, kernel support).
  Compare their flags rather than trusting one.
- Save the preprocessor and threshold with the model so `predict` reproduces the run exactly.

## Next

`../04_time_series/` switches to ordered data, where the split must respect time.