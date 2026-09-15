# gradient boosting classifier — worked mathematics

02_math_intuition.cpp
Lesson: one softmax round of GB, by hand.
With two classes (log-odds z), the link from scores to probabilities is the
softmax/sigmoid, and the boosting residual is exactly y - p:
  p0 = 1/(1+exp(-z0)),  r_i = y_i - p_i.
A single pass shows why "probability residual" is the right target: the
model pushes probability mass exactly where it is missing.
sklearn equivalent: sklearn.ensemble.GradientBoostingClassifier.

Worked values: two rows with log-odds z = [0.5, -0.5], labels y=[1,0].

EXPECTED OUTPUT:
  z = [0.5, -0.5]
  p0 = sigmoid(0.5) = 0.6224593   p1 = sigmoid(-0.5) = 0.3775407
  CE = 0.474077
  pseudo-residual r = [0.3775407, -0.3775407]
  next tree reduces exactly this residual



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
