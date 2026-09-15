# decision tree classifier — worked mathematics

02_math_intuition.cpp
Lesson: choosing a split by Gini reduction.
Four rows have feature x=[1,2,3,4] and class y=[0,0,1,1].  The best split
maximises the drop in impurity:
  gain = G(parent) - (nL/n)*G(left) - (nR/n)*G(right).
Try the two inner thresholds; x <= 2.5 separates perfectly, others waste a
split - exactly what the greedy search reports back.
sklearn equivalent: sklearn.tree.DecisionTreeClassifier(criterion='gini').

EXPECTED OUTPUT:
  parent Gini 0.5   (2/4 vs 2/4)
  split x<=1.5 : G 0 + 0.4444444 -> weighted 0.3333333  gain 0.1666667
  split x<=2.5 : G 0 + 0        -> weighted 0  gain 0.5
  best split x <= 2.5  (gain 0.5 = perfect separation)



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
