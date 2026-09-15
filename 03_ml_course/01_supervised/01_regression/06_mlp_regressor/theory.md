# mlp regressor — theory

01_theory.cpp
Lesson: the multi-layer perceptron in one page.
Theory summary:
  - A neuron computes  h = act(w . x + b); layers stack them:
      input x -> linear + ReLU -> hidden -> linear -> output yhat.
  - Regression: last layer is linear, loss = mean squared error.
  - Training uses BACKPROPAGATION: the chain rule computes dLoss/dw for
    every weight, then Adam takes a step downhill.  Repeat for epochs.
  - A network with one hidden layer can fit ANY continuous function
    (universal approximation) - depth buys efficiency, not just power.
sklearn equivalent: sklearn.neural_network.MLPRegressor.

Tiny forward pass: x=1, w=0.5, b=0 (identity output, no hidden layer).



## Statistical assumptions and model choice

Observations must represent the population on which predictions will be used. Correlated subjects or time-ordered observations require grouped or temporal splits. A flexible model can fit training noise: choose complexity using training-only validation, then report a separate holdout once. Compare with a constant or majority baseline. Check feature units, missingness, duplicate observations and target availability before modeling.

## Learning sequence

Read math_intuition.md, run the first three numbered lessons, inspect the implementation functions, then run the final end-to-end lesson. The small lessons isolate one calculation; the final project combines preprocessing, model selection, evaluation and persistence.
