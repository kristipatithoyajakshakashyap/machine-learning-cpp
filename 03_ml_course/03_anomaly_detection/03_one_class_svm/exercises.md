# One-Class SVM: exercises

1. **Hand computation with three points.** Take `x = {0, 1, 4}`, `gamma = 1`, `nu = 1` (so `C = 1/3` and every alpha is bounded). Compute the kernel matrix, `g = K alpha`, rho and the score of the query `x = 2` by hand, then confirm with `OneClassSVM(1, 1)`.

2. **Sweep gamma.** In `03_implementation.cpp` fit with gamma in {0.05, 0.5, 5, 50} and export the decision grid for each. Plot or inspect the fraction of grid cells inside the region. Explain why the very large gamma model flags almost every grid cell even though the training flag rate stays near nu.

3. **The nu property.** Fit the blob from `03_implementation.cpp` with nu in {0.01, 0.05, 0.1, 0.25, 0.5} and, for each, print the number of support vectors, the number of bounded support vectors (alpha at C) and the training flag rate at threshold 0. Verify the two inequalities from `theory.md` hold for every nu.

4. **Compare the three detectors.** Score the penguin test partition with Isolation Forest, LOF and the one-class SVM using the same split and the same 95th percentile validation rule. Join the flagged row ids. Which birds are flagged by all three? Which by exactly one? Look at their raw measurements and argue which method's notion of "unusual" each one reflects.

5. **Solver convergence.** Add a counter to a scratch copy of `fit` that records the KKT gap `g_up - g_down` after every sweep, and write it to CSV for the penguin training set. Does the gap decrease monotonically? How many sweeps are needed for tolerances 1e-3, 1e-6 and 1e-9, and do the held-out flags change between them?
