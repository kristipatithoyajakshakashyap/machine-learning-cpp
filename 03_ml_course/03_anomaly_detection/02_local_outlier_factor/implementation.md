# Local Outlier Factor: implementation walkthrough

`LocalOutlierFactor.hpp` is header-only in namespace `ml` and depends only on `02_unsupervised/validation.hpp` (for `validate_dense` and `distance2`) and `helper/persistence/archive.hpp`.

## State

| member | meaning |
|---|---|
| `k_requested_` | k given to the constructor |
| `k_` | effective k after clamping to n - 1 at `fit` |
| `threshold_` | flag threshold (default 1.5) |
| `train_` | preprocessed training matrix, stored for novelty scoring |
| `kdist_`, `lrd_` | per-training-row k-distance and local reachability density |
| `lof_` | outlier-mode factors of the training rows (`fitted_scores()`) |

## fit(const Mat& X)

1. `validate_dense(X)` rejects empty, ragged or non-finite input; fewer than two rows or k = 0 throw `invalid_argument`.
2. Copy X into `train_` and clamp `k_ = min(k, n - 1)`.
3. For every row i, `neighbours(X[i], i)` computes the distance to all other rows, `std::partial_sort`s the k closest (ties broken by index, so results are deterministic) and records the k-th distance as `kdist_[i]`.
4. With all k-distances known, `density()` computes `lrd_[i] = k / sum max(kdist_[o], d(i, o))`.
5. With all densities known, `factor()` computes `lof_[i] = mean(lrd_[o]) / lrd_[i]`.

Each pass depends on the previous one, so the neighbour lists are kept in a temporary vector of n * k entries until the third pass finishes.

## score_samples(const Mat& Q) (novelty mode)

For each query the k nearest *training* rows are found without exclusion, then the same `density()` and `factor()` helpers run with the stored `kdist_` and `lrd_`. Nothing about the query batch itself influences a score, which is what makes the fresh-process reload test meaningful.

## Threshold and predict

`set_threshold` accepts any positive finite value (LOF is unbounded above, unlike the (0, 1) score of Isolation Forest). `predict` returns 1 when `score > threshold`, 0 otherwise, and accepts an optional override threshold like the forest.

## Complexity

- Fit: n distance computations of cost d for each of n rows, plus a partial sort: O(n^2 d + n^2 log k). With n = 200 penguins and d = 4 this is instant; with n = 10^5 it is not, which is why production implementations use ball trees or approximate neighbours.
- Score: O(n d) per query.
- Memory: the model stores the full training matrix, so `model_state.txt` grows linearly with training size.

## The epsilon guard

`density()` divides k by `max(sum_reach, 1e-12)`. Reachability sums are zero only when every neighbour is at distance 0 and has k-distance 0, i.e. at least k + 1 identical rows. Without the guard lrd would be infinite and the LOF ratio NaN; with it, identical rows get the same large density and a factor of exactly 1. The unit test `epsilon guard on zero distances` fits four identical rows and checks the query score is finite.

## Persistence

`save` writes the tag `"LOF_V1"`, `k_`, `threshold_`, `train_`, `kdist_` and `lrd_` through `archive::write` (17 significant digits, so doubles round-trip exactly). `load` reads them back and rejects a wrong tag, empty training data, an inconsistent k or mismatched vector lengths. Because the neighbour search is deterministic and the stored doubles are exact, a reloaded model reproduces `score_samples` bit for bit; `04_end_to_end.cpp` asserts this through `unsup::verify` and the `ulof_fresh_reload` ctest repeats it in a separate process.

## End-to-end flow (`04_end_to_end.cpp`)

Load penguins, seed a 60/20/20 shuffle, run EDA on the training partition, fit the `Preprocessor` on training rows only, fit LOF with k = 20, score validation rows and take their 95th percentile as the threshold, score and flag the test rows, snapshot model and preprocessor, reload and verify, then write predictions, partition assignments, metrics, manifest, figure and report.
