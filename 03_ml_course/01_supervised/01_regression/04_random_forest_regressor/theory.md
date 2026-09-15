# random forest regressor — theory

01_theory.cpp
Lesson: random forest regression in one page.
Theory summary:
  - A forest trains n_trees independent CART trees and averages their
    predictions:  yhat = (1/T) sum_t tree_t(x).
  - Each tree sees its own BOOTSTRAP sample (n rows drawn with
    replacement), and each split considers only max_features random
    features.  Both tricks DECORRELATE the trees.
  - Averaging decorrelated learners lowers variance: Var(mean) =
    Var(tree) * (rho + (1-rho)/T), where rho is the trees' correlation.
    More trees rarely hurt; they only stop helping after a plateau.
sklearn equivalent: sklearn.ensemble.RandomForestRegressor.

Worked example: three shallow trees answer a query with different leaves.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
