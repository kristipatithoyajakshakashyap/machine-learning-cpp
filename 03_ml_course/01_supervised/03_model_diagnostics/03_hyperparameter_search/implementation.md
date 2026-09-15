# Implementation notes

## helper/eval/search.hpp

* `ParamSpace` holds `grid`: an ordered list of `(name, values)` axes. `size()` is the Cartesian product; `at(i)` decodes a linear index (first axis varying fastest) into a `Params` list of `(name, value)`.
* `grid_search(make, X, y, space, k, seed, metric, classification)` evaluates every index. `make(const Params&)` returns a model with `fit` and `predict`; `metric(y_true, y_pred)` is higher-is-better. Folds are `stratified_kfold` when `classification`, otherwise `kfold`. The shared RNG is re-seeded before each fit so stochastic models replay.
* `random_search(..., n_iter, ...)` shuffles the index list with `std::mt19937(seed)` and takes the first `n_iter`: distinct draws by construction, capped at the space size.
* `nested_cv(make, X, y, space, outer_k, inner_k, n_iter, seed, metric, classification)` runs a random search inside every outer fold (`seed + fold` for the inner sampler), refits the winner on the outer-training rows, scores it on the outer-test rows and returns per-fold rows plus `nested_mean/std` and the `naive` best score of a plain search on all rows.
* `best_row`, `params_to_string`, `param_value` are small conveniences for lessons.

## Model.hpp: TunedForest

`run_supervised` accepts one scalar per candidate, so `TunedForest(p)` maps `p` in `{0..8}` to `(max_depth, min_leaf)` with depth varying fastest (`depth = {4,8,16}[p % 3]`, `leaf = {1,2,5}[p / 3]`) and wraps `course::make_forest(100, depth, leaf)`. `predict_proba` forwards to the forest. The archive tag is `TunedForest_V1`; the index is saved before the forest so a reloaded model reports the same configuration.

## Lessons

* `03_implementation.cpp` runs all three searches on standardised wine and writes `grid_search.csv`, `random_search.csv`, `nested_cv.csv` and `search_comparison.svg` (best-so-far CV macro F1 versus evaluations, plus the nested estimate as a horizontal line).
* `04_end_to_end.cpp` calls `run_supervised(load_wine, {0..8}, TunedForest)`, then reproduces the stratified holdout (seed 42), fits the preprocessor on training rows only, and appends random search and nested CV artefacts to `validation/`. `--quick` uses 3 draws, 3 folds and 3 outer folds.

## Tests

`tests/model_test.cpp` uses a constant-predictor learner whose CV score is analytic: grid best equals brute-force max; random search draws exactly `n_iter` distinct combinations and is capped at the space size; nested CV returns one row per outer fold; `TunedForest` save/load reproduces predictions and probabilities exactly and rejects index 9.
