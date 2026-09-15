// 02_math_intuition.cpp
// Lesson: what a bootstrap sample actually gives you.
// Bootstrap = draw n rows WITH replacement from the n training rows.  The
// chance a particular row is missed in n draws is (1 - 1/n)^n, which for
// large n converges to 1/e = 0.3678794.  So every tree trains on roughly
// 63% of the rows and repeats the rest - each tree's "personality" comes
// from WHICH copies it saw.  This is the randomness that decorrelates the
// forest.
// sklearn equivalent: sklearn.ensemble.RandomForestRegressor
// (bootstrap=True, max_samples=1.0).
//
// EXPECTED OUTPUT:
//   missed-row probability (1-1/n)^n for bootstrap size n:
//   n=5    0.32768
//   n=10    0.3486784
//   n=100    0.3660323
//   n->inf 0.3678794  (1/e)
//   so each tree trains on ~63.2% of unique rows

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  std::cout << "missed-row probability (1-1/n)^n for bootstrap size n:\n";
  for (double n : {5.0, 10.0, 100.0}) {
    double miss = std::pow(1.0 - 1.0 / n, n);
    std::cout << "n=" << n << "    " << miss << "\n";
  }
  std::cout << "n->inf 0.3678794  (1/e)\n";
  std::cout << "so each tree trains on ~63.2% of unique rows\n";
  return 0;
}
