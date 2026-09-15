# adaboost classifier — worked mathematics

02_math_intuition.cpp
Lesson: the weight update table, by hand.
Start with five rows weighted 1/5.  One row is misclassified (e = 0.2).
Apply the SAMME update, then renormalise - the five new weights clearly
point the next tree at the mistake.
sklearn equivalent: sklearn.ensemble.AdaBoostClassifier (SAMME).

EXPECTED OUTPUT:
  before: all w = 0.2
  alpha = 1.386294   (e = 0.2)
  after raw update (correct *0.25, wrong *4) and normalise:
  row 0 (wrong) w = 0.8
  row 1 (right) w = 0.05
  row 2 (right) w = 0.05
  row 3 (right) w = 0.05
  row 4 (right) w = 0.05
  the wrong row now carries 16x the weight of a right row



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
