# Exercises

1. Run `diag_04_permutation_importance` and compare the ranking with the impurity-based ranking printed by `06_random_forest_classifier`. Explain one feature whose rank differs and why.
2. Duplicate the most important wine column (add it twice) in `04_permutation_importance.cpp`. Predict what happens to its importance before running, then run and explain.
3. Change `metric` from macro-F1 to accuracy in `05_learning_curve.cpp`. Which fractions change most, and why does the class-imbalanced wine target make the two metrics diverge at small sizes?
4. Set `k = 3` and `k = 10` for the learning curve. Describe how the CV curve's smoothness and its final value change, and relate it to the number of test rows per fold.
5. Replace the forest in `Forest.hpp` with a shallow decision tree (depth 2). Rerun `06_end_to_end.cpp` and diagnose the model from the learning curve alone: bias-limited or variance-limited?
