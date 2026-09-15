// 01_theory.cpp
// Lesson: support vector machine in one page.
// Theory summary:
//   - Linear SVM finds a hyperplane w.x + b = 0 with the LARGEST margin
//     (distance to the closest training points, the "support vectors").
//   - Instead of squared loss it minimises the HINGE loss
//       max(0, 1 - y*(w.x + b)) + regularization,
//     which only punishes rows inside (or across) the margin.
//   - The decision function is f(x) = sum_i alpha_i*y_i*K(x_i,x) + b, so
//     kernels (K = exp(-gamma||x-x'||^2) for RBF) let a linear solver draw
//     non-linear boundaries by working in feature space.
// sklearn equivalent: sklearn.svm.SVC (kernel='linear' / 'rbf').
//
// Margin worksheet: w=(1,0), b=0.
//
// EXPECTED OUTPUT:
//   margin distance = 1/||w|| = 1
//   point (2,1) y=+1 : margin score 2, hinge 0
//   point (0.5,1) y=+1 : margin score 0.5, hinge 0.5
//   point (2,1) y=-1 : margin score -2, hinge 3
//   only misclassified / near-boundary rows pay a loss

#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  std::cout << "margin distance = 1/||w|| = 1\n";
  double x1 = 2, x2 = 0.5;
  std::cout << "point (2,1) y=+1 : margin score " << x1 << ", hinge "
            << std::max(0.0, 1.0 - x1) << "\n";
  std::cout << "point (0.5,1) y=+1 : margin score " << x2 << ", hinge "
            << std::max(0.0, 1.0 - x2) << "\n";
  std::cout << "point (2,1) y=-1 : margin score " << -x1 << ", hinge "
            << std::max(0.0, 1.0 + x1) << "\n";
  std::cout << "only misclassified / near-boundary rows pay a loss\n";
  return 0;
}
