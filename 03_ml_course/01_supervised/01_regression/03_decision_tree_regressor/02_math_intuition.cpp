// 02_math_intuition.cpp
// Lesson: why a split is worth taking - variance reduction.
// A regression tree split "buys" a drop in variance.  We measure a parent
// node's spread, try a threshold, and accept the split when the weighted
// child variance is clearly smaller:
//   reduction = var(parent) - (nL/n)*var(left) - (nR/n)*var(right)
// With y = x + 1 (perfectly linear) the x <= 2 cut traps the two left rows
// in a tight bucket - most of the variance is already explained by one rule.
// This is the same "impurity reduction" idea as Gini for classifiers.
// sklearn equivalent: sklearn.tree.DecisionTreeRegressor (criterion='squared_error').
//
// EXPECTED OUTPUT:
//   x = [1 2 3 4 5]  y = [2 3 4 5 6]
//   var(parent) 2
//   split x <= 2 : left mean 2.5 var 0.25 (2 rows)  right mean 5 var 0.6666667 (3 rows)
//   weighted child variance 0.5
//   variance reduction 1.5

#include <iomanip>
#include <iostream>

#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);

  ml::Vec y = {2, 3, 4, 5, 6};
  std::cout << "x = [1 2 3 4 5]  y = [2 3 4 5 6]\n";
  std::cout << "var(parent) " << ml::variance(y) << "\n";

  ml::Vec left = {2, 3};
  ml::Vec right = {4, 5, 6};
  std::cout << "split x <= 2 : left mean " << ml::mean(left) << " var "
            << ml::variance(left) << " (" << left.size()
            << " rows)  right mean " << ml::mean(right) << " var "
            << ml::variance(right) << " (" << right.size() << " rows)\n";

  double w = left.size() / 5.0 * ml::variance(left) +
             right.size() / 5.0 * ml::variance(right);
  std::cout << "weighted child variance " << w << "\n";
  std::cout << "variance reduction " << ml::variance(y) - w << "\n";
  return 0;
}
