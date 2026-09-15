// 01_theory.cpp
// Lesson: Gaussian Naive Bayes in one page.
// Theory summary:
//   - Bayes: P(y|x) = P(y) * f(x|y) / sum_y' P(y')*f(x|y'), where f is the
//     Gaussian density with the class mean/variance.
//   - "Naive": each feature is assumed independent, so the likelihood is the
//     product over features - a strong assumption that still works well with
//     few rows and little data.
//   - Classification rule: pick the class with the largest posterior.  For
//     numeric stability libraries compare LOG posteriors instead.
// sklearn equivalent: sklearn.naive_bayes.GaussianNB.
//
// Toy data: class 0 rows x={1,2,3}, class 1 rows x={7,8,9}, query x=4.5.
// priors = 0.5 each, variances = 0.6666667 each.
//
// EXPECTED OUTPUT:
//   priors 0.5 0.5
//   f(x=4.5|class0, mu=2, var=0.6666667) = 0.004499875
//   f(x=4.5|class1, mu=8, var=0.6666667) = 4.99891e-05
//   posterior class0 = 0.9890131
//   posterior class1 = 0.01098695
//   predicted class 0

#include <cmath>
#include <iomanip>
#include <iostream>

double gauss(double x, double mu, double var) {
  return std::exp(-(x - mu) * (x - mu) / (2.0 * var)) /
         std::sqrt(2.0 * std::acos(-1.0) * var);
}

int main() {
  std::cout << std::setprecision(7);
  double p0 = 0.5, p1 = 0.5;
  double x = 4.5;
  double f0 = gauss(x, 2.0, 0.6666667);
  double f1 = gauss(x, 8.0, 0.6666667);
  std::cout << "priors 0.5 0.5\n";
  std::cout << "f(x=4.5|class0, mu=2, var=0.6666667) = " << f0 << "\n";
  std::cout << "f(x=4.5|class1, mu=8, var=0.6666667) = " << f1 << "\n";
  double den = p0 * f0 + p1 * f1;
  double post0 = p0 * f0 / den;
  std::cout << "posterior class0 = " << post0 << "\n";
  std::cout << "posterior class1 = " << p1 * f1 / den << "\n";
  std::cout << "predicted class " << (post0 > 0.5 ? 0 : 1) << "\n";
  return 0;
}
