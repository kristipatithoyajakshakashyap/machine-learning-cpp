# One-Class SVM: implementation walkthrough

`OneClassSVM.hpp` is header-only in namespace `ml` and depends only on `02_unsupervised/validation.hpp` (for `validate_dense` and `distance2`) and `helper/persistence/archive.hpp`.

## State

| member | meaning |
|---|---|
| `nu_` | outlier / support-vector budget in (0, 1] |
| `gamma_requested_` | gamma given to the constructor; `<= 0` means "scale" (`1 / p`) |
| `gamma_` | effective gamma resolved at `fit` |
| `rho_` | offset of the separating hyperplane |
| `threshold_` | flag threshold on the score (default 0 = the boundary) |
| `support_` | preprocessed training rows with positive alpha |
| `alpha_` | their dual coefficients |

## fit(const Mat& X)

1. `validate_dense(X)` rejects empty, ragged or non-finite input; fewer than two rows or nu outside (0, 1] throw `invalid_argument`.
2. Resolve `gamma_` and build the symmetric `n x n` RBF kernel matrix `K`.
3. Start from the feasible point `alpha_i = 1 / n` and compute `g = K alpha`.
4. Pairwise descent for at most `kMaxSweeps` sweeps: choose the most violating pair (`up`: largest `g` with `alpha > 0`; `down`: smallest `g` with `alpha < C`), stop when `g_up - g_down < kKktTolerance`, otherwise move the clipped exact step and update `g` in O(n).
5. Collect rows with `alpha > 0` into `support_` / `alpha_`. `rho_` is the mean of `g` over free support vectors (`0 < alpha < C`), or over all support vectors if none is free.

Ties are broken by row index in both selections, so training is deterministic.

## decision_function / score_samples

`decision_function(Q)` returns `sum_i alpha_i K(s_i, q) - rho` for every query. `score_samples(Q)` negates it so that higher means more anomalous and zero is the boundary, consistent with `IsolationForest` and `LocalOutlierFactor`. Queries are validated against the support-vector dimension.

## Threshold and predict

`set_threshold(v)` accepts any finite value (scores can be negative, so zero is not a sentinel). `predict(Q, threshold = NaN)` uses the stored threshold when NaN is passed and returns 1 for `score > threshold`, else 0.

## save / load

`archive::write` stores the tag `OneClassSVM_V1`, `nu`, effective `gamma`, `rho`, `threshold`, the support vectors and their alphas. `load` rejects a wrong tag, empty or mismatched support data, non-positive gamma, nu outside (0, 1] and non-finite rho or threshold, then re-validates the support matrix. Because scoring uses only the stored quantities, a reloaded model reproduces `score_samples` bit for bit.

## The end-to-end project

`04_end_to_end.cpp` mirrors the LOF workflow: seeded 60/20/20 split, `Preprocessor` fitted on training rows only, `OneClassSVM(0.05)` with scale gamma, validation-score 95th percentile threshold, held-out scoring, `unsup::snapshot` of model and preprocessing, in-process reload check with `unsup::verify`, then the standard artifacts (`evaluation/predictions.csv`, `metrics.json`, `run_manifest.json`, `report.md`, `execution.log`, figures).

`predict.cpp` reloads the snapshot from the end-to-end results directory and scores a header-less numeric CSV. `02_unsupervised/tests/verify_reload.cpp` (branch `ocsvm`) recomputes the held-out scores in a fresh process and compares them with the saved CSV.

## Numerical notes

- `curvature = K_uu + K_dd - 2 K_ud` is clamped to `1e-12` so duplicate rows cannot divide by zero.
- `kMaxSweeps = 20000` bounds the solver; on the course data sets it converges in far fewer sweeps.
- Training cost is dominated by the kernel matrix (O(n^2 p)) and one O(n) update per sweep.
