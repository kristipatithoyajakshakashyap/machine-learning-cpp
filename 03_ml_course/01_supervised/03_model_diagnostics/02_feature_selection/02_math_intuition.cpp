#include "helper/eval/feature_selection.hpp"
#include <iomanip>
#include <iostream>
// ANOVA F by hand on a two-class fixture, then the helper's answer.
int main() {
  std::cout
      << std::setprecision(4)
      << R"LESSON(ANOVA F = (between-class variance / (k-1)) / (within-class variance / (n-k)). A column whose class means differ a lot relative to its spread inside each class scores high. |Pearson r| plays the same role for a continuous target. Neither sees interactions or redundancy, which is what wrappers and lasso add.)LESSON"
      << "\n\n";
  ml::Mat X = {{1, 10}, {2, 20}, {3, 30}, {7, 10}, {8, 20}, {9, 30}};
  ml::Vec y = {0, 0, 0, 1, 1, 1};
  // Column 0: class means 2 and 8, grand mean 5.
  const double between = 3 * 9 + 3 * 9;         // n_c (mean_c - grand)^2
  const double within = 2 + 2;                  // (x - mean_c)^2 summed
  const double f0 = (between / 1) / (within / 4);
  std::cout << "column 0 by hand: between " << between << " within " << within
            << " F = " << f0 << "\n";
  const ml::Vec scores = ml::filter_scores(X, y, true);
  std::cout << "filter_scores: " << ml::print_vec(scores)
            << " (column 1 has identical class means, F = 0)\n";
  const ml::Vec r = ml::filter_scores(X, {1, 2, 3, 4, 5, 6}, false);
  std::cout << "|Pearson r| against a linear target: " << ml::print_vec(r)
            << "\n";
}
