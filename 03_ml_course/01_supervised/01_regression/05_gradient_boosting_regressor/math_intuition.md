# gradient boosting regressor — worked mathematics

02_math_intuition.cpp
Lesson: the residual table, by hand.
Boosting is easiest to trust with three rows and no trees in the way:
start from the constant mean, subtract, fit the residuals exactly, shrink,
subtract again.  Two rounds already walk the ensemble toward y; the
absolute residual shrinks from 2 to 1 to 0.
sklearn equivalent: sklearn.ensemble.GradientBoostingRegressor(lr=0.5).

EXPECTED OUTPUT:
  row    y   F after 0   r0        F after 1   r1        F after 2   r2
  0      2   4           -2        3           -1        2.5           -0.5
  1      4   4           0        4           0        4           0
  2      6   4           2        5           1        5.5           0.5
  with lr=0.5 and h = residuals exactly



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
