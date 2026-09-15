// 01_theory.cpp
// Lesson: decision tree classification in one page.
// Theory summary:
//   - A tree classifies by a chain of "X[f] <= t" tests; leaves hold a class
//     distribution and answer with the most common class.
//   - Splits are chosen to reduce GINI IMPURITY
//       G = 1 - sum_c p_c^2   (0 = pure leaf, max when classes mix).
//   - max_depth / min_samples_leaf control how far the recursion goes - the
//     classic bias-variance dial of every tree model.
// sklearn equivalent: sklearn.tree.DecisionTreeClassifier(criterion='gini').
//
// Toy leaf: 3 green, 1 red (classes {0,0,0,1}).
//
// EXPECTED OUTPUT:
//   parent leaf Gini = 0.375   (p = 0.75, 0.25)
//   perfect split -> Gini 0
//   no split -> 0.375
//   trees keep splitting until Gini hits 0 or depth runs out

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double g = 1.0 - (0.75 * 0.75 + 0.25 * 0.25);
  std::cout << "parent leaf Gini = " << g << "   (p = 0.75, 0.25)\n";
  std::cout << "perfect split -> Gini 0\n";
  std::cout << "no split -> 0.375\n";
  std::cout << "trees keep splitting until Gini hits 0 or depth runs out\n";
  return 0;
}
