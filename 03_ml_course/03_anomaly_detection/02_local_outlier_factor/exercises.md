# Local Outlier Factor: exercises

1. **Hand computation with k = 3.** Redo the worked example in `math_intuition.md` (points 0, 1, 2, 10) with k = 3 instead of k = 2. Every point now has all three others as neighbours. Compute the k-distances, reachability distances, densities and factors by hand, then confirm them by fitting `LocalOutlierFactor(3)` and printing `fitted_scores()`. Explain why LOF(D) shrinks when k grows.

2. **Between-cluster points.** In `03_implementation.cpp` the planted outlier at (3, 3) lies between the two clusters. Score the same synthetic data with `IsolationForest` from the previous module and compare the ranks of the three planted outliers under both methods. Which one is easiest for LOF and hardest for the forest, and why?

3. **Sweep k.** Modify `04_end_to_end.cpp` (or write a small driver) to fit LOF with k in {5, 10, 20, 40, 80} on the penguin training partition, keep the 95th percentile validation threshold, and record the test flag rate and the set of flagged row ids for each k. How stable is the flagged set? Which rows are flagged at every k?

4. **Epsilon guard.** Build a training set with exactly k + 1 identical rows plus a few distinct ones and score one more identical row. Trace which reachability distances are zero, verify the score is finite and equal to 1, then remove the `max(sum, 1e-12)` guard in a scratch copy of the header and observe what happens.

5. **Inspect the flags.** Open `results/04_end_to_end_results/evaluation/predictions.csv`, join the flagged row ids with the original penguin records via `data/row_assignments.csv`, and look at the raw measurements. Do the flagged birds have extreme values in one feature, or unusual combinations? Argue why species labels cannot be used to compute precision for these flags.
