// 01_theory.cpp
// Lesson: neural network classification in one page.
// Theory summary:
//   - Classification nets replace the linear output with a SOFTMAX head:
//       p_c = exp(z_c) / sum_k exp(z_k),   approx p = 1/(1+exp(-z)) binary.
//   - Loss = cross-entropy; backpropagation delivers dL/dz = p - y (one-hot)
//     straight to the last layer, and the chain rule pushes gradients into
//     every hidden weight.
//   - SGD/Adam + epochs is the whole trainer.  A hidden layer with
//     ReLU activations plus a softmax head = logistic regression with
//     engineered (learned) features.
// sklearn equivalent: sklearn.neural_network.MLPClassifier.
//
// Two-class head: log-odds difference z0 - z1 = 2.
//
// EXPECTED OUTPUT:
//   z0 - z1 = 2
//   p0 = 1/(1+exp(-2)) = 0.8807971
//   true label 0 -> CE = 0.126928

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double z0 = 1.5, z1 = -0.5;
  double p0 = std::exp(z0) / (std::exp(z0) + std::exp(z1));
  std::cout << "z0 - z1 = 2\n";
  std::cout << "p0 = 1/(1+exp(-2)) = " << p0 << "\n";
  std::cout << "true label 0 -> CE = " << -std::log(p0) << "\n";
  return 0;
}
