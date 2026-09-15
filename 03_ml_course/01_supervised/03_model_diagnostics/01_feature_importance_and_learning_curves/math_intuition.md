# Mathematical intuition

## Permutation importance

Let `f` be a fitted model, `(X, y)` a scoring set with `n` rows and `p` columns, and `S(y, f(X))` a metric where larger is better (macro-F1 here). Define the baseline

    s0 = S(y, f(X)).

For column `j`, draw a random permutation `pi` of `{1..n}` and build `X^(j,pi)` by replacing column `j` with its permuted values while every other column is untouched. The permuted score is

    s_j(pi) = S(y, f(X^(j,pi))).

The importance is the expected drop, estimated over `R` repeats:

    I_j = s0 - (1/R) * sum_r s_j(pi_r),
    sd_j = standard deviation of the R drops.

Shuffling preserves the marginal distribution of column `j` (same values, same histogram) but destroys its joint relation with `y` and with the other columns. If the model never looks at column `j`, `f(X^(j,pi)) = f(X)` and `I_j = 0` exactly. Negative values are sampling noise around zero.

### Worked example (3 rows, 2 features)

Model: `f(x) = 1 if x1 > 0 else 0`, ignoring `x2`. Data `x1 = (-1, 2, 3)`, `x2 = (5, 6, 7)`, `y = (0, 1, 1)`. Accuracy baseline `s0 = 1`.

- Permute `x2` in any way: predictions unchanged, `I_2 = 0`.
- Permute `x1` with `pi = (2, 3, 1)`: `x1 = (2, 3, -1)`, predictions `(1, 1, 0)`, accuracy `1/3`, drop `2/3`. Averaging over all 6 permutations gives `I_1 = 1 - 5/9 = 4/9`.

`02_math_intuition.cpp` reproduces this average numerically.

## Learning curve

Choose fractions `f_1 < ... < f_m` in `(0, 1]` and `k` folds. For each fold `(train_k, test_k)` and fraction `f`, take the first `ceil(f * |train_k|)` rows of a seeded shuffle of `train_k` (stratified by class so every class survives small fractions), fit a fresh model on that subset, and record

    train_score(f, k) = S(y_sub, f_hat(X_sub)),
    cv_score(f, k)    = S(y_test_k, f_hat(X_test_k)).

Report the fold means for each `f`, plus `n_train = ceil(f * |train_k|)`.

### Reading the two curves

Write the expected test error as `bias^2 + variance + noise`. Increasing `n_train` shrinks the variance term roughly like `1/n` for smooth learners and leaves bias unchanged. Therefore:

- `cv_score` still rising at `f = 1` means variance dominates; more rows will help.
- `cv_score` flat and far below `train_score` means the model memorises but the remaining gap is not closing; regularise or simplify.
- both flat and close means the model is bias-limited; change the hypothesis class or features.

The training score of a random forest sits near 1.0 for all sizes because each tree fits its bootstrap sample exactly; use the CV curve for decisions.

## Complexity

Permutation importance costs `p * R` extra prediction passes. The learning curve costs `m * k` fits. Both are embarrassingly parallel but are run sequentially here so the seeded RNG replay is exact.
