# mlp regressor — worked mathematics

02_math_intuition.cpp
Lesson: one backpropagation step, by hand.
For a single linear neuron with loss L = (yhat - y)^2:
  dL/dw = 2*(yhat-y)*x ,  dL/db = 2*(yhat-y),  and Adam/SGD walks
  w <- w - lr*dL/dw.  The gradient points uphill; stepping against it
  lowers the loss.  Deeper nets repeat this exact chain rule per layer.
sklearn equivalent: the "solver" inside
sklearn.neural_network.MLPRegressor.

Worked step: x=1, w=0.5, b=0, y=2, lr=0.1.

EXPECTED OUTPUT:
  yhat = 0.5  loss = 2.25
  dL/dw = -3  dL/db = -3
  w <- 0.8  b <- 0.3
  new yhat = 1.1  new loss = 0.81



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
