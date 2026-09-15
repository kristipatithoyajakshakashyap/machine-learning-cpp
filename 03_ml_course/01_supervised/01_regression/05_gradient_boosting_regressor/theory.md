# gradient boosting regressor — theory

01_theory.cpp
Lesson: gradient boosted trees in one page.
Theory summary:
  - GB builds an ADDITIVE model: F(x) = F0 + lr*h1(x) + lr*h2(x) + ...
    each new tree h fits what the ensemble still gets wrong.
  - For squared error the "what is still wrong" is exactly the residual
    r_i = y_i - F(x_i); for other losses it is the negative gradient of
    the loss at F, one number per row.
  - lr (shrinkage) scales every tree; 0.1 works far better than 1.0 by
    letting each weak tree take a small step.
  - n_estimators + lr + max_depth are the whole model: too many round
    trees overfit (boosters are easy to overfit).
sklearn equivalent: sklearn.ensemble.GradientBoostingRegressor.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
