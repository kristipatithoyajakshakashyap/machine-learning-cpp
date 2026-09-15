// 01_theory.cpp
// Lesson: gradient boosted trees in one page.
// Theory summary:
//   - GB builds an ADDITIVE model: F(x) = F0 + lr*h1(x) + lr*h2(x) + ...
//     each new tree h fits what the ensemble still gets wrong.
//   - For squared error the "what is still wrong" is exactly the residual
//     r_i = y_i - F(x_i); for other losses it is the negative gradient of
//     the loss at F, one number per row.
//   - lr (shrinkage) scales every tree; 0.1 works far better than 1.0 by
//     letting each weak tree take a small step.
//   - n_estimators + lr + max_depth are the whole model: too many round
//     trees overfit (boosters are easy to overfit).
// sklearn equivalent: sklearn.ensemble.GradientBoostingRegressor.
//
// EXPECTED OUTPUT:
//   y = [2 4 6]
//   F0 = mean(y) = 4
//   step 1 residuals r = y - F0 = [-2 0 2]
//   h1 = residuals, F1 = F0 + lr*h1  (lr=0.5) -> [3 4 5]
//   step 2 residuals r = [-1 0 1]
//   each tree chases a smaller and smaller error

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double lr = 0.5;
  std::cout << "y = [2 4 6]\n";
  std::cout << "F0 = mean(y) = 4\n";
  std::cout << "step 1 residuals r = y - F0 = [-2 0 2]\n";
  std::cout << "h1 = residuals, F1 = F0 + lr*h1  (lr=" << lr << ") -> [3 4 5]\n";
  std::cout << "step 2 residuals r = [-1 0 1]\n";
  std::cout
      << "each tree chases a smaller and smaller error\n";
  return 0;
}
