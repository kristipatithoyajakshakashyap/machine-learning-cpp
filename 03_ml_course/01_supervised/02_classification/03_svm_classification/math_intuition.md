# svm classification — worked mathematics

02_math_intuition.cpp
Lesson: hinge loss and the RBF kernel, by hand.
Two numbers carry all of SVM:
  1. hinge(z) = max(0, 1 - y*z): positive only inside the margin.
  2. K(x,x') = exp(-gamma * ||x-x'||^2): "how alike are these rows",
     from 1 (identical) down to ~0 (far apart).
A kernel SVM answers with f(x) = sum alpha_i*y_i*K(x_i,x) + b, so these
two formulas ARE the classifier.
sklearn equivalent: sklearn.svm.SVC(kernel='rbf').

Worked values: w-plane score z = w.x with w=(1,0); RBF with gamma=0.1.

EXPECTED OUTPUT:
  hinge(1 - 2) = 0              (confident correct)
  hinge(1 - 0.5) = 0.5          (inside the margin)
  hinge(1 - (-2)) = 3           (misclassified)
  K((0,0),(3,4)) = exp(-0.1*25) = 0.082085
  K((0,0),(1,0)) = exp(-0.1*1) = 0.9048374



## Derivation exercise

Reproduce the displayed quantities using the loops in 02_math_intuition.cpp. Identify the dimensions of every vector and matrix, the reduction over samples, and any regularization parameter. Change one observation and calculate the new result by hand before rerunning.

## Generalization

Training error measures fit to observations used by the optimizer. Validation error selects hyperparameters; test error estimates performance after selection. Fitting a transformation is part of model fitting, so its parameters must be recomputed inside each training fold.
