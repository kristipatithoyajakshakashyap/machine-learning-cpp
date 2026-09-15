# decision tree regressor — worked mathematics

02_math_intuition.cpp
Lesson: why a split is worth taking - variance reduction.
A regression tree split "buys" a drop in variance.  We measure a parent
node's spread, try a threshold, and accept the split when the weighted
child variance is clearly smaller:
  reduction = var(parent) - (nL/n)*var(left) - (nR/n)*var(right)
With y = x + 1 (perfectly linear) the x <= 2 cut traps the two left rows
in a tight bucket - most of the variance is already explained by one rule.
This is the same "impurity reduction" idea as Gini for classifiers.
sklearn equivalent: sklearn.tree.DecisionTreeRegressor (criterion='squared_error').

EXPECTED OUTPUT:
  x = [1 2 3 4 5]  y = [2 3 4 5 6]
  var(parent) 2
  split x <= 2 : left mean 2.5 var 0.25 (2 rows)  right mean 5 var 0.6666667 (3 rows)
  weighted child variance 0.5
  variance reduction 1.5



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
