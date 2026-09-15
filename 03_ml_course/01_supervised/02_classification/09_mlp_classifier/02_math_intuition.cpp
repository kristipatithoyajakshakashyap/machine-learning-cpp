// 02_math_intuition.cpp
// Lesson: softmax and one backpropagation step, by hand.
// For a softmax head with cross-entropy, the gradient at the logits is the
// famous one-liner
//       dL/dz_k = p_k - y_k     (y one-hot),
// proof being a short chain-rule exercise.  Everything below the last layer
// reuses those dL/dz values via the chain rule - the entire backpropagation
// algorithm.  Here we only compute the head itself.
// sklearn equivalent: the output layer of
// sklearn.neural_network.MLPClassifier.
//
// Three logits z = [2, 1, 0.5], true class 0 (y = [1 0 0]).
//
// EXPECTED OUTPUT:
//   softmax p = [0.6285317, 0.2312239, 0.1402444]
//   cross-entropy = 0.4643688
//   dL/dz = p - y = [-0.3714683, 0.2312239, 0.1402444]
//   gradient pushes logit 0 up, the others down

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double z[3] = {2, 1, 0.5};
  double e[3] = {std::exp(z[0]), std::exp(z[1]), std::exp(z[2])};
  double s = e[0] + e[1] + e[2];
  double p[3] = {e[0] / s, e[1] / s, e[2] / s};
  std::cout << "softmax p = [" << p[0] << ", " << p[1] << ", " << p[2]
            << "]\n";
  std::cout << "cross-entropy = " << -std::log(p[0]) << "\n";
  std::cout << "dL/dz = p - y = [" << p[0] - 1.0 << ", " << p[1] << ", "
            << p[2] << "]\n";
  std::cout << "gradient pushes logit 0 up, the others down\n";
  return 0;
}
