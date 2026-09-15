// 03_ml_course/03_anomaly_detection/03_one_class_svm/02_math_intuition.cpp
// Purpose: the one-class SVM primal, dual and decision function written out,
//          then a two-point hand example (x = 0 and x = 2, gamma = 1, nu = 1)
//          checked numerically against the implementation.
// Inputs:  none (points and parameters are hard-coded).
// Outputs: prints only; throws (non-zero exit) if rho or any score deviates
//          from the hand result by more than 1e-12.
// Run target: uocsvm_math_intuition.
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "OneClassSVM.hpp"
int main() {
  std::cout
      << R"LESSON(Primal: min 1/2 ||w||^2 + 1/(nu n) sum xi_i - rho   s.t. w.phi(x_i) >= rho - xi_i, xi_i >= 0
Dual:   min 1/2 sum_ij a_i a_j K(x_i, x_j)   s.t. 0 <= a_i <= 1/(nu n), sum_i a_i = 1
Decision: f(x) = sum_i a_i K(x_i, x) - rho, with rho = sum_j a_j K(x_j, x_i) for any free support vector (0 < a_i < 1/(nu n)).
RBF kernel: K(x, z) = exp(-gamma ||x - z||^2).
Hand example, 1-D points x_0 = 0, x_1 = 2, gamma = 1, nu = 1:
  C = 1/(nu n) = 1/2, and sum a = 1 forces a_0 = a_1 = 1/2 (both at the bound).
  k = K(x_0, x_1) = exp(-4) = 0.018316
  g_0 = g_1 = (1 + k) / 2 = 0.509158  -> rho = 0.509158 (no free vector; mean over support vectors)
  f(x_0) = 0.509158 - rho = 0 (training points sit on the boundary)
  f(1)   = (exp(-1) + exp(-1)) / 2 - rho = 0.367879 - 0.509158 = -0.141279
  score(1) = rho - raw = +0.141279: the midpoint is outside because gamma = 1 makes each point its own blob.)LESSON"
      << '\n';
  // nu = 1 pins both multipliers at the bound 1/2, so everything is closed form.
  ml::Mat X{{0}, {2}};
  ml::OneClassSVM m(1, 1);
  m.fit(X);
  const double k = std::exp(-4.0), rho = (1 + k) / 2,
               expected_mid = rho - std::exp(-1.0);
  // Score the two training points (expect 0) and the midpoint x = 1.
  const auto s = m.score_samples({{0}, {2}, {1}});
  std::cout << "rho computed " << m.rho() << " expected " << rho << '\n';
  std::cout << "score(x_0) " << s[0] << " score(x_1) " << s[1]
            << " score(1) computed " << s[2] << " expected " << expected_mid
            << '\n';
  // All four quantities are closed form, so a 1e-12 tolerance is appropriate.
  if (std::abs(m.rho() - rho) > 1e-12 || std::abs(s[0]) > 1e-12 ||
      std::abs(s[1]) > 1e-12 || std::abs(s[2] - expected_mid) > 1e-12)
    throw std::runtime_error("hand computation mismatch");
  std::cout << "Numerical check passed: implementation matches the hand "
               "computation.\n";
}
