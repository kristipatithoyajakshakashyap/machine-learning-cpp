// 03_ml_course/01_supervised/01_regression/09_gaussian_process/02_math_intuition.cpp
// Lesson 02: numbers behind the GP prior and the marginal likelihood.
//
// Reads:   nothing (5 hard-coded points).
// Writes:  nothing (prints only).
// Run:     cmake --build --preset course --target gp_math_intuition
//          then build/03_ml_course/01_supervised/01_regression/09_gaussian_process/
//          gp_math_intuition
//
// WHAT YOU LEARN:
//   - Prior correlation exp(-d^2 / (2 l^2)) between inputs a distance d apart
//     for length scales 0.3, 1, 3, 10.
//   - How the log marginal likelihood of the same 5 points changes with the
//     length scale: the Occam trade-off between data fit and model complexity.
//
// EXPECTED OUTPUT:
//   One explanatory paragraph, four "correlation at d=1 ..., d=3 ..." lines and
//   four "log marginal likelihood" lines (4 decimals).
#include "Model.hpp"
#include <iomanip>
#include <iostream>
// Print-only: how the kernel length scale controls correlation between
// inputs, and how the marginal likelihood trades fit against complexity.
// Correlation table: closed form, no model needed.
int main() {
  std::cout
      << R"LESSON(Two inputs a distance d apart have prior correlation exp(-d^2 / (2 l^2)). A short length scale makes distant points independent (wiggly functions, high effective complexity); a long one couples everything (smooth functions). The log marginal likelihood -y'K^-1 y/2 - log|K|/2 - n log(2pi)/2 rewards fitting the data (first term) and penalises kernels that could explain many datasets (second term), so its maximiser is an Occam compromise. Every fit is O(n^3) because of the Cholesky factorisation.)LESSON"
      << "\n";
  std::cout << std::fixed << std::setprecision(4);
  for (double l : {0.3, 1.0, 3.0, 10.0}) {
    std::cout << "length_scale " << l << ": correlation at d=1 "
              << std::exp(-0.5 / (l * l)) << ", d=3 " << std::exp(-4.5 / (l * l)) << "\n";
  }
  // y is roughly sin(x); noise_var 0.01 keeps K well conditioned.
  ml::Mat X = {{0}, {1}, {2}, {3}, {4}};
  ml::Vec y = {0, 0.84, 0.91, 0.14, -0.76};
  for (double l : {0.3, 1.0, 3.0, 10.0}) {
    course::GaussianProcess gp(l, 1.0, 0.01);
    gp.fit(X, y);
    std::cout << "length_scale " << l << " log marginal likelihood "
              << gp.log_marginal_likelihood() << "\n";
  }
}
