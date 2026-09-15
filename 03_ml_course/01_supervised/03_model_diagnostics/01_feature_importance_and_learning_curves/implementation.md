# Implementation notes

All code is in `helper/eval/diagnostics.hpp` (header-only) and reused by lessons 03 to 06 and by the tests.

## Data structures

    struct ImportanceRow      { size_t feature; double mean_drop; double std_drop; };
    struct LearningCurvePoint { size_t n_train; double train_score; double cv_score; };

Both are plain aggregates so they can be written straight to CSV.

## permutation_importance(model, X, y, metric, repeats, seed)

1. Validate: non-empty `X`, matching `y`, `repeats > 0`, otherwise throw `std::invalid_argument`.
2. Compute the baseline `metric(y, model.predict(X))` once.
3. `std::mt19937 rng(seed)`; for every column `j`:
   - copy `X` once into a working matrix;
   - for each repeat, draw a permutation of row indices with `std::shuffle`, overwrite column `j` from the original matrix in permuted order, predict, and store `baseline - metric`;
   - accumulate mean and standard deviation of the drops.
4. Return one row per feature in column order; the lessons sort by `mean_drop` for display.

The model is passed by const reference and never refitted, so the function works for any type exposing `Vec predict(const Mat&) const`.

## learning_curve(make, X, y, fractions, k, seed, metric, stratified)

1. Validate: `k >= 2`, every fraction in `(0, 1]`.
2. Build folds with `stratified_kfold` or `kfold` from `helper/eval/cross_validation.hpp` using `seed`.
3. `subset_rows(rows, fraction, seed, y, stratified)` returns the first `ceil(fraction * count)` rows of a seeded shuffle; when stratified the quota is applied per class and the result is sorted so row order is deterministic.
4. For each fraction and fold: `auto m = make(); m.fit(X_sub, y_sub);` then score on the subset and on the fold's test rows.
5. Average over folds and return one point per fraction.

Before every `fit` the shared course RNG (`ml::rng()`) is re-seeded from `seed`, so stochastic models such as the random forest replay identically across runs and across `--quick` and full modes.

## Forest.hpp

`course::make_forest(n_trees, max_depth = 8, min_leaf = 2)` returns the course `ml::RandomForest` from `01_regression/04_random_forest_regressor` switched to `TreeTask::Classification`. The algorithm is compiled once into the shared `ml_tree_models` library, so this module links that and needs no library of its own.

## 06_end_to_end.cpp

Runs `run_supervised` with the tree-count grid `{20, 50, 100}`. After the pipeline returns 0 and inference mode is not requested, it refits the selected forest on the training rows, computes permutation importance with 10 repeats (macro-F1) and the learning curve with a 50-tree forest, and writes `evaluation/permutation_importance.csv` and `evaluation/figures/learning_curve.svg` into `full/` or `quick/`.

## Numerical guards

- Metric drops are clamped to finite values; a NaN metric throws.
- A feature whose column is constant produces a drop of exactly 0 because the permuted column equals the original.
- `subset_rows` never returns fewer rows than the number of classes when stratified.
