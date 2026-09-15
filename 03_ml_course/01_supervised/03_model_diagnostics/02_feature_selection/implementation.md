# Implementation notes

The selection algorithms are in `helper/eval/feature_selection.hpp` (header-only, namespace `ml`) so other modules can call them. `Model.hpp` adds `course::SelectedForest`. The module library `ml_fsel` is an INTERFACE target linking `ml_tree_models` (the forest) and `ml_rlin` (the lasso used by `l1_path`).

## Helper API

    Vec filter_scores(const Mat& X, const Vec& y, bool classification);
    std::vector<size_t> top_k(const Vec& scores, size_t k);
    std::vector<ForwardStep> forward_selection(Factory make, const Mat& X, const Vec& y,
        size_t k_max, size_t folds, unsigned seed, Metric metric, bool stratified = true);
    Mat l1_path(const Mat& X, const Vec& y, const Vec& lambdas);
    Vec l1_scores(const Mat& path, const Vec& lambdas);

* `filter_scores` validates that `X` is rectangular, finite and matches `y`; ANOVA F for classification, |Pearson r| otherwise. A constant column scores 0; a column with zero within-class scatter but separated means scores +inf.
* `top_k` returns indices of the `k` largest scores with a stable sort, so ties go to the lower index (deterministic across platforms).
* `forward_selection` takes a factory returning any model with `fit(X, y)` and `predict(X)`, and a higher-is-better metric such as `ml::macro_f1`. The folds are built once from `seed`; `seed_rng(seed)` is called before each fit so stochastic models replay exactly.
* `l1_path` standardizes `X` once and calls `LinearRegression::fit_lasso` for every lambda; row `l` of the result holds the slopes (intercept dropped) at `lambdas[l]`. `l1_scores` turns the path into one rank score per feature: the largest lambda at which it is non-zero, with a small tie-breaker from |coef| at the smallest lambda.

## SelectedForest

    class SelectedForest {
      size_t k_;                 // how many columns to keep
      bool classification_;
      size_t n_features_;        // width seen at fit(); predict() checks it
      std::vector<size_t> kept_; // column indices, best first
      ml::Vec scores_;           // filter score of every column
      ml::RandomForest forest_;  // 50 trees via course::make_forest
    };

The constructor takes `double k` so `run_supervised` can pass `{5, 10, 20, 30}` directly; non-integer or non-positive `k` throws. `fit` scores the training rows, keeps the top `k`, projects and fits the forest. `predict` and `predict_proba` project new rows onto `kept_` after checking the width. `save` writes the tag `SelectedForest_V1`, then `k_`, `classification_`, `n_features_`, `kept_`, `scores_`, then the forest's own state; `load` checks the tag and that the kept indices are consistent before reading the forest. The unit test checks that a reloaded model reproduces probabilities bit for bit.

## Lessons

* `01_theory`: a six-row fixture where only column 0 separates the classes; prints the kept column and its scores.
* `02_math_intuition`: ANOVA F by hand versus `filter_scores`, then |Pearson r|.
* `03_implementation`: filter, forward (8 steps, 3-fold macro F1, 20-tree forest) and lasso rankings on the breast cancer training split. Writes `rankings.csv`, `rankings.svg` (filter rank versus lasso rank) and `l1_path.csv`.
* `04_end_to_end`: `run_supervised` with `SelectedForest` over `k in {5, 10, 20, 30}`, then `validation/filter_scores.csv`, `validation/forward_selection.csv` and `validation/figures/forward_selection.svg` computed from the same stratified 80/20 training split the pipeline used (seed 42). `--quick` uses a smaller forest and 4 forward steps.

## Tests (`tests/model_test.cpp`)

Synthetic data with a strong signal column, a mild one and a noise column: the filter ranks them in that order; forward selection picks the signal column first with CV accuracy above 0.95; the lasso keeps the signal column longest; `SelectedForest(2)` keeps columns {0, 1}, fits, and survives save/load exactly; `k` larger than the width and empty data both throw.
