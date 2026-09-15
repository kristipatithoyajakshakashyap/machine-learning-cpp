// 01_theory.cpp
// Lesson: gradient boosted classification in one page.
// Theory summary:
//   - GB regression fits trees to residuals; classification fits trees to
//     PSEUDO-RESIDUALS of the logistic loss.  For binary:
//       r_i = y_i - p_i,   p_i = sigmoid(F(x_i)).
//   - Models are summed in LOG-ODDS space, then piped through the sigmoid
//     (or softmax for K classes) to get probabilities:
//       p = sigmoid( F0(x) + lr*h1(x) + lr*h2(x) + ... )
//   - Leaf values are set so the round actually minimises the loss, then
//     everything repeats.  This is why the same boosting loop covers both
//     regression and classification.
// sklearn equivalent: sklearn.ensemble.GradientBoostingClassifier.
//
// Binary worksheet: y=[1,0], current p = [0.6224593, 0.3775407].
//
// EXPECTED OUTPUT:
//   pseudo-residual r = y - p = [0.3775407, -0.3775407]
//   the classifier's next tree chases probability error, not raw labels

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double y[2] = {1, 0};
  double p[2] = {0.6224593, 0.3775407};
  std::cout << "pseudo-residual r = y - p = ["
            << y[0] - p[0] << ", " << y[1] - p[1] << "]\n";
  std::cout
      << "the classifier's next tree chases probability error, not raw labels\n";
  return 0;
}
