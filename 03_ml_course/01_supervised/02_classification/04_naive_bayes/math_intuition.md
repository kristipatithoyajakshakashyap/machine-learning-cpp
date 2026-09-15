# naive bayes — worked mathematics

02_math_intuition.cpp
Lesson: posteriors in log space - the same math, without underflow.
Multiplying many tiny probabilities (a product over features, over rows of
evidence) underflows to zero on real hardware.  The fix: work with
  logP(y|x) = log P(y) + sum_f log f(x_f | y)
and rebuild the probabilities with the softmax trick - subtract the max of
the two log scores, then exp.  The two-class posterior is just a sigmoid:
  P(class0|x) = 1 / (1 + exp(logP1 - logP0)).
sklearn equivalent: sklearn.naive_bayes.GaussianNB (log-spaced by default).

Same toy as theory: class 0 x={1,2,3}, class 1 x={7,8,9}, query x=8.

EXPECTED OUTPUT:
  query x=8
  log likelihood  class0 = -28.40935
  log likelihood  class1 = -1.409353
  log posterior class0 = -28.40935  class1 = -1.409353
  posterior class0 = 1.879531e-12
  posterior class1 = 1
  predicted class 1 (same decision as regular Bayes, safe numbers)



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
