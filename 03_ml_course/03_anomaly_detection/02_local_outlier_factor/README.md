# Local Outlier Factor

## Method and core idea

Local Outlier Factor (Breunig et al., 2000) asks whether a row is less dense than its own
neighbours, not whether it is far from everything. For each row the k nearest neighbours are
found. The reachability distance to a neighbour is the larger of the true distance and that
neighbour's k-distance (which smooths out noise inside dense groups). The local reachability
density (lrd) is the inverse of the mean reachability distance. The factor is the mean ratio of
the neighbours' lrd to the row's own lrd. A factor near 1 means "as dense as my neighbourhood".
A factor well above 1 marks an outlier, even if the row would look normal on a global
distance scale. This makes LOF the right tool when clusters have different spreads. In novelty
mode the training rows and their densities are stored, and new rows are scored against them
without changing the stored statistics. Cost is O(n^2 p) for the brute-force neighbour search
used here.

## Dataset and why

Palmer Penguins with a seeded 60/20/20 split and k = 20. The species form groups of different
spread, which is exactly where a global distance threshold would be misleading. No anomaly labels
exist, so only factors and flag rates are reported. The implementation lesson uses two synthetic
2-D clusters of different spread with three planted outliers.

## Prerequisites

Nearest neighbours and standardization (`../../02_unsupervised/`), the Isolation Forest protocol
in `../01_isolation_forest/`. Read `theory.md`, `math_intuition.md` and `implementation.md`, then
`exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `LocalOutlierFactor.hpp` | interface library `ml_local_outlier_factor` | Header-only `ml::LocalOutlierFactor`: k-distance, reachability, lrd, `fit`, `fitted_scores`, `score_samples` (novelty), `set_threshold`/`threshold`, `predict`, `save`/`load`. k is clamped to n - 1 | - |
| `01_theory.cpp` | `ulof_theory` | Lesson text plus a 1-D demonstration with two groups of different density and one stray point | prints only |
| `02_math_intuition.cpp` | `ulof_math_intuition` | k-distance, reachability, lrd and LOF worked by hand on four points (k = 2) and verified against the class to 1e-12 | prints only |
| `03_implementation.cpp` | `ulof_implementation` | Two 2-D clusters of different spread plus three planted outliers. Top factors, flags at LOF > 1.5, novelty scoring demo | `results/03_implementation_results/{scores.csv, figures/lof_scatter.svg, figures/lof_histogram.svg}` |
| `04_end_to_end.cpp` | `ulof_end_to_end` | Full project: 60/20/20 split, training-only EDA and preprocessing, fit (k = 20, novelty), validation-calibrated threshold, test scoring, snapshot, reload check | `results/04_end_to_end_results/` |
| `predict.cpp` | `ulof_predict` | Reloads preprocessor, stored training rows and threshold. Writes a factor and a flag per new row | `results/predict_results/new_predictions.csv` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target ulof_end_to_end ulof_predict
build\03_ml_course\03_anomaly_detection\02_local_outlier_factor\ulof_end_to_end.exe
build\03_ml_course\03_anomaly_detection\02_local_outlier_factor\ulof_predict.exe new_rows.csv
```

No `--quick` flag. `ulof_predict <numeric_csv>` takes a header-less CSV in the order of
`model/feature_schema.csv`, raw units, blanks/`NA`/`nan` allowed. Output: `output_0` = local
outlier factor, `output_1` = flag using the saved threshold.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv`, `eda/` (training rows), `validation/{calibration_scores.csv,
threshold.json}` (validation factors and the 95th-percentile threshold),
`evaluation/{predictions.csv, metrics.json, figures/scores.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` (the model file stores the
training rows with their k-distances and densities) and `inference/reload_verification.json`. The
`ulof_fresh_reload` test writes `results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R ulof` runs `ulof_workflow`, `ulof_fresh_reload` (recompute factors for
the saved test rows in a new process and compare with `evaluation/predictions.csv`) and
`ulof_new_rows`. `ctest --preset course -R unsupervised_numerical` includes the LOF checks in
`../../02_unsupervised/tests/test_unsupervised.cpp`: a far point scores above 2 and outranks every
cluster member, novelty scoring flags only the far point, save/load reproduces k, threshold and
scores, k is clamped to n - 1, and zero distances stay finite.

## Key takeaways

- LOF is relative: the same distance is normal in a sparse group and anomalous in a tight one.
- k sets the neighbourhood scale. Too small is noisy, too large blurs local structure.
- Novelty mode freezes the training statistics, which is what makes `predict` reproducible.

## Next

`../03_one_class_svm/` learns an explicit boundary around the normal data in kernel space.