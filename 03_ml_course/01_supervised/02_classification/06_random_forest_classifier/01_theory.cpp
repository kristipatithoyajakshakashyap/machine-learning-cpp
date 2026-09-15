// 01_theory.cpp
// Lesson: random forest classification in one page.
// Theory summary:
//   - Same recipe as the regressing forest: T bootstrap trees, each split
//     over a random subset of features, predictions combined.
//   - For classification a tree emits CLASS PROBABILITIES (the leaf's class
//     shares); the forest averages a probability vector:
//         P(y=c|x) = (1/T) sum_t p_t(c|x).
//   - Averaging probabilities is smoother than a hard vote and agrees with
//     sklearn's predict_proba.  Hard votes = argmax of that average.
// sklearn equivalent: sklearn.ensemble.RandomForestClassifier.
//
// Three trees answer a two-class query with probability vectors.
//
// EXPECTED OUTPUT:
//   tree predictions: [0.9 0.1] [0.8 0.2] [0.4 0.6]
//   forest average: [0.7 0.3]  -> class 0
//   hard vote would be 2 votes vs 1 -> still class 0

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double p[3][2] = {{0.9, 0.1}, {0.8, 0.2}, {0.4, 0.6}};
  double avg0 = (p[0][0] + p[1][0] + p[2][0]) / 3.0;
  std::cout << "tree predictions: [0.9 0.1] [0.8 0.2] [0.4 0.6]\n";
  std::cout << "forest average: [" << avg0 << " " << (1.0 - avg0)
            << "]  -> class 0\n";
  std::cout << "hard vote would be 2 votes vs 1 -> still class 0\n";
  return 0;
}
