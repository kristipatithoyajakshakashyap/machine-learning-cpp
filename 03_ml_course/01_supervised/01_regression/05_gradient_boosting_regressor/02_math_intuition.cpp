// 02_math_intuition.cpp
// Lesson: the residual table, by hand.
// Boosting is easiest to trust with three rows and no trees in the way:
// start from the constant mean, subtract, fit the residuals exactly, shrink,
// subtract again.  Two rounds already walk the ensemble toward y; the
// absolute residual shrinks from 2 to 1 to 0.
// sklearn equivalent: sklearn.ensemble.GradientBoostingRegressor(lr=0.5).
//
// EXPECTED OUTPUT:
//   row    y   F after 0   r0        F after 1   r1        F after 2   r2
//   0      2   4           -2        3           -1        2.5           -0.5
//   1      4   4           0        4           0        4           0
//   2      6   4           2        5           1        5.5           0.5
//   with lr=0.5 and h = residuals exactly

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double F0 = 4, lr = 0.5;
  double y[] = {2, 4, 6};
  std::cout << "row    y   F after 0   r0        F after 1   r1        F after 2   r2\n";
  for (int i = 0; i < 3; ++i) {
    double F1 = F0 + lr * (y[i] - F0);
    double F2 = F1 + lr * (y[i] - F1);
    std::cout << i << "      " << y[i] << "   " << F0 << "           "
              << (y[i] - F0) << "        " << F1 << "           "
              << (y[i] - F1) << "        " << F2 << "           "
              << (y[i] - F2) << "\n";
  }
  std::cout << "with lr=" << lr << " and h = residuals exactly\n";
  return 0;
}
