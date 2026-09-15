# knn regression — worked mathematics

02_math_intuition.cpp
Lesson: distance is the whole model - the math behind kNN.
A kNN model has nothing to learn: it stores the training rows and answers
each query by measuring Euclidean distance to every stored point, keeping
the k smallest, and averaging their targets.  Smaller k = tighter fit to
the training set (low bias, high variance); larger k = smoother (high bias,
low variance).
sklearn equivalent: sklearn.neighbors.KNeighborsRegressor with
metric='minkowski', p=2 (Euclidean), weights='uniform'.

Worked example: query q=(3,3); neighbours A=(1,1) y=10, B=(5,2) y=20,
C=(2,6) y=30.

EXPECTED OUTPUT:
  d(A) 2.828427
  d(B) 2.236068
  d(C) 3.162278
  k=1 -> B -> 20
  k=2 -> mean(20,10) = 15
  k=3 -> mean(20,10,30) = 20



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
