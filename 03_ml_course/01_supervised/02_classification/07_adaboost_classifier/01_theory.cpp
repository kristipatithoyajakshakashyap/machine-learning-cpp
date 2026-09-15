// 01_theory.cpp
// Lesson: AdaBoost in one page.
// Theory summary:
//   - AdaBoost trains weak trees SEQUENTIALLY on the same rows, but each
//     round re-weights the sample: rows the previous trees got wrong carry
//     more weight, so the next tree must focus on them.
//   - Round t with weighted error e:
//       alpha_t = ln((1-e)/e) + ln(n_classes-1)
//     down-weights correct rows by exp(-alpha) and up-weights errors by
//     exp(+alpha), then all weights are normalised back.
//   - Final answer = weighted vote, weights alpha_t.  The ensemble is
//     adaptive: no two rounds have the same goals.
// sklearn equivalent: sklearn.ensemble.AdaBoostClassifier (SAMME).
//
// Weighted-error worksheet, 2-class round with e = 0.2.
//
// EXPECTED OUTPUT:
//   weighted error e = 0.2
//   alpha = ln(4) = 1.386294
//   correct rows get w *= exp(-alpha) = 0.25
//   wrong rows get   w *= exp(+alpha) = 4
//   next round must fix the hard row

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double e = 0.2;
  double alpha = std::log((1.0 - e) / e);
  std::cout << "weighted error e = 0.2\n";
  std::cout << "alpha = ln(" << (1.0 - e) / e << ") = " << alpha << "\n";
  std::cout << "correct rows get w *= exp(-alpha) = " << std::exp(-alpha)
            << "\n";
  std::cout << "wrong rows get   w *= exp(+alpha) = " << std::exp(alpha)
            << "\n";
  std::cout << "next round must fix the hard row\n";
  return 0;
}
