# Exercises

1. Extend the `ParamSpace` in `03_implementation.cpp` with a third axis `n_trees {50, 100, 200}` (change `make`). How many evaluations does the grid need now? Run random search with 9 draws and compare its best-so-far curve to the 27-point grid.
2. Set `k = 2` in `grid_search` and re-run five times with seeds 1..5. Record the winning `(max_depth, min_leaf)` per seed. Explain the instability using the sqrt(2 ln m) argument in math_intuition.md.
3. In `nested_cv.csv`, compare `naive` to `nested_mean` on wine, then repeat with `ml::subsample(wine, 60, 1)`. Does the gap grow as rows shrink? Why?
4. Implement a `halving_search` that evaluates all candidates with 2 folds, keeps the top half, and re-evaluates survivors with more folds. Compare its budget and its winner to `grid_search`.
5. Why does `random_search` sample without replacement here, and when would sampling continuous distributions (as sklearn's `RandomizedSearchCV` does) be the better choice?
