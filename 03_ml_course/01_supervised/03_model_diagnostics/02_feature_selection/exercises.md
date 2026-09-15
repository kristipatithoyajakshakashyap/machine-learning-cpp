# Exercises

1. Run `fsel_implementation` and open `rankings.csv`. Find a feature whose filter rank is in the top 5 but whose lasso rank is below 15. Using `l1_path.csv`, explain in terms of correlated columns why the lasso dropped it.
2. In `04_end_to_end.cpp`, move the call to `filter_scores` so it runs on the whole dataset instead of the training split, then compare `validation/candidate_scores.csv` before and after. Why does the CV curve change even though `SelectedForest::fit` itself is unchanged?
3. Change the forward-selection metric from `ml::macro_f1` to `ml::accuracy` and rerun. Which steps change? Relate the answer to the class balance of breast cancer.
4. Add a `backward_elimination` helper next to `forward_selection` that starts from all columns and removes the one whose removal hurts CV least. Run both on the training split with `k = 8` and compare the chosen sets.
5. Replace the lambda grid in `03_implementation.cpp` with a geometric grid of 30 values from 1 down to 1e-4. Plot how many features are non-zero at each lambda. At what lambda does the count reach the pipeline's selected `k`?
6. Make `SelectedForest` accept a mode flag that selects with `l1_scores` instead of `filter_scores` (fit a linear-probability lasso on 0/1 labels). Save/load must still round-trip; extend the unit test to prove it.
