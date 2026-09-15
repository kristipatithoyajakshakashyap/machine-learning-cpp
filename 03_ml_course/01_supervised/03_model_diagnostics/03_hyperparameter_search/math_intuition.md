# Math intuition: optimistic bias and nested cross-validation

## Best of m noisy scores

Suppose m candidate configurations have identical true quality mu, and each CV mean is mu + e_i with e_i ~ N(0, sigma^2) independent. Reporting max_i (mu + e_i) reports mu + max_i e_i, whose expectation is positive and grows roughly like sigma * sqrt(2 ln m). Nine candidates with fold noise sigma = 0.02 already inflate the reported macro F1 by about 0.04 when in fact nothing was gained. The bias is worse when candidates are many, folds few and the dataset small.

## Why the CV winner's score is biased

The CV score played two roles: it *selected* the winner and it *reports* the winner. The rows used for reporting are the same rows that influenced the choice, so they are no longer held out with respect to the whole procedure (search + fit).

## Nested cross-validation

Split the data into outer folds. For each outer fold, run the *entire* search (inner CV over candidates) on the outer-training part only, refit the selected candidate on that part, and score once on the outer-test part. The outer-test rows never influenced any decision, so the mean outer score is an honest estimate of the performance of *the procedure* "search, then fit".

Note what nested CV estimates: not a single model, but the pipeline. Different outer folds may select different parameters; that variability is itself informative (a stable choice means the search is not chasing noise).

## Random search hit probability

If a fraction q of the space is "good enough", n independent draws all miss it with probability (1 - q)^n. Sixty draws hit a top-5% region with probability 1 - 0.95^60, about 0.95, independent of dimensionality. The `hps_math_intuition` lesson prints this table.

## Reading `nested_cv.csv`

`naive` is the best inner CV score obtained when the search sees all rows: the optimistic number. `nested_mean` is the mean outer score. The gap is the selection bias on this dataset; on wine it is typically small because the forest is insensitive to the grid, which is itself a useful finding.
