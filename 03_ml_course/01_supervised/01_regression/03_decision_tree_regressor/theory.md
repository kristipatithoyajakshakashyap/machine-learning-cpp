# decision tree regressor — theory

01_theory.cpp
Lesson: CART regression trees in one page.
Theory summary:
  - A classification/regression tree (CART) splits rows by "X[feature] <= t".
  - Each split is chosen to minimise the SSE of the left + right child:
      gain = SS(parent) - SS(left) - SS(right)
  - The best feature+threshold pair wins, then the process recurses on each
    child until leaves hold few rows or max_depth is reached.
  - Leaf value = mean of the rows that land there.
  - max_depth is the bias/variance dial: deep trees memorise (low train SSE,
    worse test), shallow trees underfit.
A single feature only - so the chosen split is exactly the first rule a
real regression tree would store after its root.
sklearn equivalent: sklearn.tree.DecisionTreeRegressor.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
