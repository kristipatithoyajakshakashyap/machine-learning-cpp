// 02_math_intuition.cpp
// Lesson: hinge loss and the RBF kernel, by hand.
// Two numbers carry all of SVM:
//   1. hinge(z) = max(0, 1 - y*z): positive only inside the margin.
//   2. K(x,x') = exp(-gamma * ||x-x'||^2): "how alike are these rows",
//      from 1 (identical) down to ~0 (far apart).
// A kernel SVM answers with f(x) = sum alpha_i*y_i*K(x_i,x) + b, so these
// two formulas ARE the classifier.
// sklearn equivalent: sklearn.svm.SVC(kernel='rbf').
//
// Worked values: w-plane score z = w.x with w=(1,0); RBF with gamma=0.1.
//
// EXPECTED OUTPUT:
//   hinge(1 - 2) = 0              (confident correct)
//   hinge(1 - 0.5) = 0.5          (inside the margin)
//   hinge(1 - (-2)) = 3           (misclassified)
//   K((0,0),(3,4)) = exp(-0.1*25) = 0.082085
//   K((0,0),(1,0)) = exp(-0.1*1) = 0.9048374

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  std::cout << "hinge(1 - 2) = " << std::max(0.0, 1.0 - 2.0)
            << "              (confident correct)\n";
  std::cout << "hinge(1 - 0.5) = " << std::max(0.0, 1.0 - 0.5)
            << "          (inside the margin)\n";
  std::cout << "hinge(1 - (-2)) = " << std::max(0.0, 1.0 + 2.0)
            << "           (misclassified)\n";
  double g = 0.1;
  std::cout << "K((0,0),(3,4)) = exp(-0.1*25) = " << std::exp(-g * 25.0)
            << "\n";
  std::cout << "K((0,0),(1,0)) = exp(-0.1*1) = " << std::exp(-g * 1.0) << "\n";
  return 0;
}
