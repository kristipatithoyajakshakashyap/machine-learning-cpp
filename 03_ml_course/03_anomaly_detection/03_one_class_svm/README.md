# One-Class SVM

## Method and core idea

The one-class support vector machine (Schoelkopf et al., 2001) learns a boundary around the
*normal* data only. Rows are mapped into an RBF kernel feature space, where the algorithm finds the
hyperplane w . phi(x) = rho that separates the training rows from the origin with the largest
margin. Two knobs control it: `nu` in (0, 1] is an upper bound on the fraction of training rows
that end up outside the boundary and a lower bound on the fraction of support vectors (a review
budget, like the threshold in the other modules), and `gamma` is the RBF width (larger gamma =
more wiggly boundary). The dual problem is a small quadratic programme solved here by a
deterministic SMO-style pairwise descent. The decision value of a new row is
sum_i a_i K(x_i, x) - rho: negative means outside the boundary, so the anomaly score used
throughout the module is rho minus the raw kernel sum (higher = more anomalous). Cost is O(n^2)
kernel evaluations per sweep of the solver.

## Dataset and why

Palmer Penguins with a seeded 60/20/20 split, nu = 0.05 and gamma = 1/p (p = 4 features). The
data has no anomaly labels, so the project reports scores and flag rates only. The implementation
lesson uses one synthetic 2-D blob with four planted outliers and draws the learned decision
region on a 41 x 41 grid, which makes the effect of nu and gamma visible.

## Prerequisites

Kernels and support vectors from the supervised SVM modules (`../../01_supervised/`), the
anomaly protocol in `../01_isolation_forest/`. Read `theory.md`, `math_intuition.md` and
`implementation.md`, then `exercises.md`.

## Files

| File | Target | What it does | Outputs |
|---|---|---|---|
| `theory.md`, `math_intuition.md`, `implementation.md`, `exercises.md` | - | Reading material | - |
| `OneClassSVM.hpp` | interface library `ml_one_class_svm` | Header-only `ml::OneClassSVM`: RBF kernel, SMO-style dual solver, `decision_function`, `score_samples`, `nu`/`gamma`/`rho`/`support_count`, `set_threshold`/`threshold`, `predict`, `save`/`load` | - |
| `01_theory.cpp` | `uocsvm_theory` | Lesson text (margin from the origin, meaning of nu and gamma) plus a 1-D cluster-and-stray example | prints only |
| `02_math_intuition.cpp` | `uocsvm_math_intuition` | Primal, dual and decision function, then a two-point hand example (nu = 1, gamma = 1) verified against the class to 1e-12 | prints only |
| `03_implementation.cpp` | `uocsvm_implementation` | One 2-D blob plus four planted outliers (nu = 0.1, gamma = 0.5): top scores, support vectors, training flag rate, decision grid | `results/03_implementation_results/{scores.csv, decision_grid.csv, figures/decision_boundary.svg, figures/scores_scatter.svg, figures/scores_histogram.svg}` |
| `04_end_to_end.cpp` | `uocsvm_end_to_end` | Full project: 60/20/20 split, training-only EDA and preprocessing, fit, validation-calibrated threshold, test scoring, snapshot, reload check | `results/04_end_to_end_results/` |
| `predict.cpp` | `uocsvm_predict` | Reloads preprocessor, support vectors, rho and threshold; writes a score and a flag per new row | `results/predict_results/new_predictions.csv` |

## Build and run

```powershell
cmake --preset course
cmake --build --preset course --target uocsvm_end_to_end uocsvm_predict
build\03_ml_course\03_anomaly_detection\03_one_class_svm\uocsvm_end_to_end.exe
build\03_ml_course\03_anomaly_detection\03_one_class_svm\uocsvm_predict.exe new_rows.csv
```

No `--quick` flag. `uocsvm_predict <numeric_csv>` takes a header-less CSV in the order of
`model/feature_schema.csv`, raw units, blanks/`NA`/`nan` allowed. Output: `output_0` = anomaly
score (rho minus decision value), `output_1` = flag using the saved threshold.

## Results layout

`results/04_end_to_end_results/` contains `run_manifest.json`, `report.md`, `execution.log`,
`data/row_assignments.csv`, `eda/` (training rows), `validation/{calibration_scores.csv,
threshold.json}`, `evaluation/{predictions.csv, metrics.json, figures/scores.svg}`,
`model/{model_state.txt, preprocessing_state.txt, feature_schema.csv}` (support vectors with dual
coefficients, gamma, rho and the threshold) and `inference/reload_verification.json`. The
`uocsvm_fresh_reload` test writes `results/verify_reload_results/fresh_process_verification.json`.

## Tests

`ctest --preset course -R uocsvm` runs `uocsvm_workflow`, `uocsvm_fresh_reload` (rescore the
saved test rows in a new process and compare with `evaluation/predictions.csv`) and
`uocsvm_new_rows`. `ctest --preset course -R unsupervised_numerical` includes the checks in
`../../02_unsupervised/tests/test_unsupervised.cpp`: planted outliers outrank every inlier, the
training flag rate at the boundary is close to nu, at least nu x n rows are support vectors,
save/load reproduces parameters and scores, and invalid nu or a dimension mismatch is rejected.

## Key takeaways

- nu is the budget: it bounds the training outlier fraction before you see any scores.
- gamma trades a smooth envelope against one that hugs every training point; standardize first.
- The saved model is just the support vectors and their coefficients, which is why it is small
  and reloads exactly.

## Next

`../../04_time_series/` moves to ordered data, where the split must respect time.
