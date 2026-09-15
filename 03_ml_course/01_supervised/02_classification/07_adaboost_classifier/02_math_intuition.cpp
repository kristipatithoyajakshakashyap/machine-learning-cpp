// 02_math_intuition.cpp
// Lesson: the weight update table, by hand.
// Start with five rows weighted 1/5.  One row is misclassified (e = 0.2).
// Apply the SAMME update, then renormalise - the five new weights clearly
// point the next tree at the mistake.
// sklearn equivalent: sklearn.ensemble.AdaBoostClassifier (SAMME).
//
// EXPECTED OUTPUT:
//   before: all w = 0.2
//   alpha = 1.386294   (e = 0.2)
//   after raw update (correct *0.25, wrong *4) and normalise:
//   row 0 (wrong) w = 0.8
//   row 1 (right) w = 0.05
//   row 2 (right) w = 0.05
//   row 3 (right) w = 0.05
//   row 4 (right) w = 0.05
//   the wrong row now carries 16x the weight of a right row

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double w[5] = {0.2, 0.2, 0.2, 0.2, 0.2};
  bool wrong[5] = {true, false, false, false, false};
  double e = 0.2;
  double alpha = std::log((1.0 - e) / e);
  std::cout << "before: all w = 0.2\n";
  std::cout << "alpha = " << alpha << "   (e = 0.2)\n";
  double sum = 0;
  for (int i = 0; i < 5; ++i) {
    w[i] *= wrong[i] ? std::exp(alpha) : std::exp(-alpha);
    sum += w[i];
  }
  std::cout << "after raw update (correct *0.25, wrong *4) and normalise:\n";
  for (int i = 0; i < 5; ++i)
    std::cout << "row " << i << " (" << (wrong[i] ? "wrong" : "right")
              << ") w = " << w[i] / sum << "\n";
  std::cout << "the wrong row now carries 16x the weight of a right row\n";
  return 0;
}
