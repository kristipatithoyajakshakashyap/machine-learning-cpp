// 01_theory.cpp
// Lesson: the multi-layer perceptron in one page.
// Theory summary:
//   - A neuron computes  h = act(w . x + b); layers stack them:
//       input x -> linear + ReLU -> hidden -> linear -> output yhat.
//   - Regression: last layer is linear, loss = mean squared error.
//   - Training uses BACKPROPAGATION: the chain rule computes dLoss/dw for
//     every weight, then Adam takes a step downhill.  Repeat for epochs.
//   - A network with one hidden layer can fit ANY continuous function
//     (universal approximation) - depth buys efficiency, not just power.
// sklearn equivalent: sklearn.neural_network.MLPRegressor.
//
// Tiny forward pass: x=1, w=0.5, b=0 (identity output, no hidden layer).
//
// EXPECTED OUTPUT:
//   yhat = w*x + b = 0.5
//   loss (y=2) = (0.5-2)^2 = 2.25

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double x = 1, w = 0.5, b = 0;
  double yhat = w * x + b;
  double loss = (yhat - 2.0) * (yhat - 2.0);
  std::cout << "yhat = w*x + b = " << yhat << "\n";
  std::cout << "loss (y=2) = (" << yhat << "-2)^2 = " << loss << "\n";
  return 0;
}
