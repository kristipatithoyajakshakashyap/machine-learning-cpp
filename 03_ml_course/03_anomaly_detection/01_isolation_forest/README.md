# Isolation Forest

## Method and core idea

An Isolation Forest (Liu, Ting and Zhou, 2008) grows many random binary trees. Each node picks a
random feature and a random cut between that feature's minimum and maximum in the node, until every
row is alone or the depth limit is reached. Rows that are far from the bulk get isolated after a few
splits, so a short average path length across the trees signals an anomaly. The path length is
normalised by the expected path length of a random binary search tree of the subsample size, giving
a score in (0, 1) where values near 1 are anomalous and values around 0.5 or lower are ordinary. No
distances or densities are computed, which makes the method cheap (O(trees x sample x log sample))
and robust to irrelevant features. The default here is 100 trees on 256-row subsamples, seed 42.

## Dataset and why

Palmer Penguins with a seeded 60/20/20 split. The data has no anomaly labels, so the project
demonstrates the honest protocol: fit on train, calibrate a 95th-percentile threshold on
validation scores, report scores and flag rate on the test split. The implementation lesson uses a
1-D line with one far point and two synthetic 2-D clusters with three planted outliers, whose
ranking is known.

## Prerequisites

Decision-tree splits (`../../01_supervised/`), standardization and the shared unsupervised
workflow (`../../02_unsupervised/`). Read `theory.md`, `math_intuition.md` and
`implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `IsolationForest.hpp` | interface library `ml_isolation_forest` | Header-only `ml::IsolationForest`: random trees, path-length `correction`, `score_samples`, `set_threshold`/`threshold`, `predict` (0/1 flags), `save`/`load` | - |
| `01_theory.cpp` | `uif_theory` | Prints the lesson text | prints only |
| `02_math_intuition.cpp` | `uif_math_intuition` | Expected path length c(n) and the score formula worked by hand | prints only |
| `03_implementation.cpp` | `uif_implementation` | 1-D sanity check, then two 2-D Gaussian clusters plus three planted outliers. Top scores printed | `results/03_implementation_results/{scores.csv, figures/isolation_scatter.svg, figures/isolation_histogram.svg}` |
| `04_end_to_end.cpp` | `uif_end_to_end` | Full project: 60/20/20 split, training-only EDA and preprocessing, fit, validation-calibrated threshold, test scoring, snapshot, reload check | `results/04_end_to_end_results/` |
| `predict.cpp` | `uif_predict` | Reloads preprocessor, forest and threshold. Writes a score and a flag per new row | `results/predict_results/new_predictions.csv` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target uif_end_to_end uif_predict
build\03_ml_course\03_anomaly_detection\01_isolation_forest\uif_end_to_end.exe
build\03_ml_course\03_anomaly_detection\01_isolation_forest\uif_predict.exe new_rows.csv
```

No `--quick` flag. `uif_predict <numeric_csv>` takes a header-less CSV in the order of
`model/feature_schema.csv`, raw units, blanks/`NA`/`nan` allowed. Output: `output_0` = anomaly
score, `output_1` = flag (1 when the score exceeds the saved threshold).
`../../02_unsupervised/tests/penguin_inference.csv` is an example input.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv`, `eda/` (training rows), `validation/{calibration_scores.csv,
threshold.json}`, `evaluation/{predictions.csv, metrics.json, figures/scores.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` and
`inference/reload_verification.json`. The `uif_fresh_reload` test writes
`results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R uif` runs `uif_workflow`, `uif_fresh_reload` (rescore the saved test
rows in a new process and compare with `evaluation/predictions.csv`) and `uif_new_rows`.
`ctest --preset course -R unsupervised_numerical` includes the forest checks in
`../../02_unsupervised/tests/test_unsupervised.cpp`: a far point has a shorter path (higher
score) than an inlier, and save/load reproduces both the threshold and the scores.

## Key takeaways

- Isolation, not distance: anomalies are "few and different", so they separate in few splits.
- The score is comparable across datasets thanks to the c(n) normalisation.
- The threshold is a review budget chosen on validation rows, not a truth learned from labels.

## Next

`../02_local_outlier_factor/` scores rows relative to the density of their own neighbourhood.