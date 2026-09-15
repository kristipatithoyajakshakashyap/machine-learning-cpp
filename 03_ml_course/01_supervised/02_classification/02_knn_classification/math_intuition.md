# knn classification — worked mathematics

02_math_intuition.cpp
Lesson: voting and the tie-break, by hand.
Classes are counted among the k nearest rows; the predicted class is the
one with the most votes.  What if it is a TIE?  Real libraries resolve it
by index order - np.argmax(np.bincount(...)) returns the SMALLEST class
index among the tied leaders.  This lesson walks two tie scenarios so the
rule is unambiguous.
sklearn equivalent: sklearn.neighbors.KNeighborsClassifier
(ties broken by class order).

EXPECTED OUTPUT:
  q=(0,0)  neighbours (1,0) c2, (0,1) c1, (1,1) c0
  distances: 1 1 1.414214 
  k=1: classes c2,c1 tie at distance 1 -> smallest class index -> 1
  k=3: counts {0:1,1:1,2:1} all tie -> smallest class index -> 0



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
