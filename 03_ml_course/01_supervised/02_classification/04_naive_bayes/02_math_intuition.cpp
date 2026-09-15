// 02_math_intuition.cpp
// Lesson: posteriors in log space - the same math, without underflow.
// Multiplying many tiny probabilities (a product over features, over rows of
// evidence) underflows to zero on real hardware.  The fix: work with
//   logP(y|x) = log P(y) + sum_f log f(x_f | y)
// and rebuild the probabilities with the softmax trick - subtract the max of
// the two log scores, then exp.  The two-class posterior is just a sigmoid:
//   P(class0|x) = 1 / (1 + exp(logP1 - logP0)).
// sklearn equivalent: sklearn.naive_bayes.GaussianNB (log-spaced by default).
//
// Same toy as theory: class 0 x={1,2,3}, class 1 x={7,8,9}, query x=8.
//
// EXPECTED OUTPUT:
//   query x=8
//   log likelihood  class0 = -28.40935
//   log likelihood  class1 = -1.409353
//   log posterior class0 = -28.40935  class1 = -1.409353
//   posterior class0 = 1.879531e-12
//   posterior class1 = 1
//   predicted class 1 (same decision as regular Bayes, safe numbers)

#include <cmath>
#include <iomanip>
#include <iostream>

double log_gauss(double x, double mu, double var) {
  return -0.5 * std::log(2.0 * std::acos(-1.0) * var) -
         (x - mu) * (x - mu) / (2.0 * var);
}

int main() {
  std::cout << std::setprecision(7);
  double x = 8;
  double l0 = log_gauss(x, 2.0, 0.6666667) + std::log(0.5);
  double l1 = log_gauss(x, 8.0, 0.6666667) + std::log(0.5);
  std::cout << "query x=8\n";
  std::cout << "log likelihood  class0 = " << l0 << "\n";
  std::cout << "log likelihood  class1 = " << l1 << "\n";
  double m = std::max(l0, l1);
  double w0 = std::exp(l0 - m), w1 = std::exp(l1 - m);
  std::cout << "log posterior class0 = " << l0 << "  class1 = " << l1
            << "\n";
  std::cout << "posterior class0 = " << w0 / (w0 + w1) << "\n";
  std::cout << "posterior class1 = " << w1 / (w0 + w1) << "\n";
  std::cout
      << "predicted class 1 (same decision as regular Bayes, safe numbers)\n";
  return 0;
}
