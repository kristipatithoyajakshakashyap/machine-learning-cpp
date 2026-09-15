# svm classification — theory

01_theory.cpp
Lesson: support vector machine in one page.
Theory summary:
  - Linear SVM finds a hyperplane w.x + b = 0 with the LARGEST margin
    (distance to the closest training points, the "support vectors").
  - Instead of squared loss it minimises the HINGE loss
      max(0, 1 - y*(w.x + b)) + regularization,
    which only punishes rows inside (or across) the margin.
  - The decision function is f(x) = sum_i alpha_i*y_i*K(x_i,x) + b, so
    kernels (K = exp(-gamma||x-x'||^2) for RBF) let a linear solver draw
    non-linear boundaries by working in feature space.
sklearn equivalent: sklearn.svm.SVC (kernel='linear' / 'rbf').

Margin worksheet: w=(1,0), b=0.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
