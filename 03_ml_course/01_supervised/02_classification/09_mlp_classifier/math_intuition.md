# mlp classifier — worked mathematics

02_math_intuition.cpp
Lesson: softmax and one backpropagation step, by hand.
For a softmax head with cross-entropy, the gradient at the logits is the
famous one-liner
      dL/dz_k = p_k - y_k     (y one-hot),
proof being a short chain-rule exercise.  Everything below the last layer
reuses those dL/dz values via the chain rule - the entire backpropagation
algorithm.  Here we only compute the head itself.
sklearn equivalent: the output layer of
sklearn.neural_network.MLPClassifier.

Three logits z = [2, 1, 0.5], true class 0 (y = [1 0 0]).

EXPECTED OUTPUT:
  softmax p = [0.6285317, 0.2312239, 0.1402444]
  cross-entropy = 0.4643688
  dL/dz = p - y = [-0.3714683, 0.2312239, 0.1402444]
  gradient pushes logit 0 up, the others down



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
