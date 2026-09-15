# mlp classifier — theory

01_theory.cpp
Lesson: neural network classification in one page.
Theory summary:
  - Classification nets replace the linear output with a SOFTMAX head:
      p_c = exp(z_c) / sum_k exp(z_k),   approx p = 1/(1+exp(-z)) binary.
  - Loss = cross-entropy; backpropagation delivers dL/dz = p - y (one-hot)
    straight to the last layer, and the chain rule pushes gradients into
    every hidden weight.
  - SGD/Adam + epochs is the whole trainer.  A hidden layer with
    ReLU activations plus a softmax head = logistic regression with
    engineered (learned) features.
sklearn equivalent: sklearn.neural_network.MLPClassifier.

Two-class head: log-odds difference z0 - z1 = 2.



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
