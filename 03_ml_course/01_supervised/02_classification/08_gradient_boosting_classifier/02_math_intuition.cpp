// 02_math_intuition.cpp
// Lesson: one softmax round of GB, by hand.
// With two classes (log-odds z), the link from scores to probabilities is the
// softmax/sigmoid, and the boosting residual is exactly y - p:
//   p0 = 1/(1+exp(-z0)),  r_i = y_i - p_i.
// A single pass shows why "probability residual" is the right target: the
// model pushes probability mass exactly where it is missing.
// sklearn equivalent: sklearn.ensemble.GradientBoostingClassifier.
//
// Worked values: two rows with log-odds z = [0.5, -0.5], labels y=[1,0].
//
// EXPECTED OUTPUT:
//   z = [0.5, -0.5]
//   p0 = sigmoid(0.5) = 0.6224593   p1 = sigmoid(-0.5) = 0.3775407
//   CE = 0.474077
//   pseudo-residual r = [0.3775407, -0.3775407]
//   next tree reduces exactly this residual

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double z0 = 0.5, y0 = 1, y1 = 0;
  double p0 = 1.0 / (1.0 + std::exp(-z0));
  double p1 = 1.0 - p0;
  std::cout << "z = [0.5, -0.5]\n";
  std::cout << "p0 = sigmoid(0.5) = " << p0 << "   p1 = sigmoid(-0.5) = "
            << p1 << "\n";
  std::cout << "CE = " << -(y0 * std::log(p0) + y1 * std::log(p1)) << "\n";
  std::cout << "pseudo-residual r = [" << (y0 - p0) << ", " << (y1 - p1)
            << "]\n";
  std::cout << "next tree reduces exactly this residual\n";
  return 0;
}
