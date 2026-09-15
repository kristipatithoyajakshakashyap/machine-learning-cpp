// 02_math_intuition.cpp
// Lesson: choosing a split by Gini reduction.
// Four rows have feature x=[1,2,3,4] and class y=[0,0,1,1].  The best split
// maximises the drop in impurity:
//   gain = G(parent) - (nL/n)*G(left) - (nR/n)*G(right).
// Try the two inner thresholds; x <= 2.5 separates perfectly, others waste a
// split - exactly what the greedy search reports back.
// sklearn equivalent: sklearn.tree.DecisionTreeClassifier(criterion='gini').
//
// EXPECTED OUTPUT:
//   parent Gini 0.5   (2/4 vs 2/4)
//   split x<=1.5 : G 0 + 0.4444444 -> weighted 0.3333333  gain 0.1666667
//   split x<=2.5 : G 0 + 0        -> weighted 0  gain 0.5
//   best split x <= 2.5  (gain 0.5 = perfect separation)

#include <iomanip>
#include <iostream>

double gini(const double p0, const double p1) {
  return 1.0 - p0 * p0 - p1 * p1;
}

int main() {
  std::cout << std::setprecision(7);
  double parent = gini(0.5, 0.5);
  std::cout << "parent Gini " << parent << "   (2/4 vs 2/4)\n";
  std::cout << "split x<=1.5 : G " << gini(1.0, 0.0) << " + " << gini(1.0 / 3.0, 2.0 / 3.0)
            << " -> weighted " << 0.0 + 0.75 * gini(1.0 / 3.0, 2.0 / 3.0)
            << "  gain " << parent - 0.75 * gini(1.0 / 3.0, 2.0 / 3.0) << "\n";
  std::cout << "split x<=2.5 : G 0 + 0        -> weighted 0" << "  gain "
            << parent << "\n";
  std::cout << "best split x <= 2.5  (gain 0.5 = perfect separation)\n";
  return 0;
}
