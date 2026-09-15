# linear regression — worked mathematics

02_gradient_descent.cpp
Lesson: instead of solving (X'X)w = X'y in one shot, we can walk downhill.
Full-batch gradient descent and Adam minimise the same MSE objective; the
closed-form OLS answer is the target they converge to. This lesson also
verifies the analytic gradient against a finite-difference check.
Equivalent: the closed form matches sklearn.linear_model.LinearRegression
on the same single feature exactly; the optimisers reproduce it by
iteration (Adam follows the Kingma & Ba update with a fixed seed).

Data: tips.csv (244 restaurant bills; total_bill -> tip). Only the
total_bill column feeds the model so all three fits share one target.

EXPECTED OUTPUT:
closed-form OLS: intercept 0.9202696  total_bill 0.1050245  R2 0.4566166
GD 1000 iters lr=1e-5: intercept 0.009094428  total_bill 0.1433519
Adam 2000 iters lr=0.2: intercept 0.9203163  total_bill 0.1050244
gradient check at w0: max|fd-analytic| = 4.615084e-09



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
