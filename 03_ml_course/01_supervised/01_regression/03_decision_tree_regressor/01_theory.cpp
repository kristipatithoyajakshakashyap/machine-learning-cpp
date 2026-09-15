// 01_theory.cpp
// Lesson: CART regression trees in one page.
// Theory summary:
//   - A classification/regression tree (CART) splits rows by "X[feature] <= t".
//   - Each split is chosen to minimise the SSE of the left + right child:
//       gain = SS(parent) - SS(left) - SS(right)
//   - The best feature+threshold pair wins, then the process recurses on each
//     child until leaves hold few rows or max_depth is reached.
//   - Leaf value = mean of the rows that land there.
//   - max_depth is the bias/variance dial: deep trees memorise (low train SSE,
//     worse test), shallow trees underfit.
// A single feature only - so the chosen split is exactly the first rule a
// real regression tree would store after its root.
// sklearn equivalent: sklearn.tree.DecisionTreeRegressor.
//
// EXPECTED OUTPUT:
//   x = [1 2 3 4 5 6]   y = [5 4 5 8 7 11]
//   SS(parent) 33.33333
//   best split threshold 3.5  left mean 4.666667 (3 rows)  right mean 8.666667 (3 rows)
//   SS(left) 0.6666667  SS(right) 8.666667  SS(children) 9.333333
//   fraction of variance explained 0.72
//   (the tree would recurse into [5 4 5] and [8 7 11] next)

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  std::cout << std::setprecision(7);

  std::vector<double> x = {1, 2, 3, 4, 5, 6};
  std::vector<double> y = {5, 4, 5, 8, 7, 11};
  std::cout << "x = [1 2 3 4 5 6]   y = [5 4 5 8 7 11]\n";

  double mean_y = 0;
  for (double v : y) mean_y += v;
  mean_y /= y.size();
  double ss_parent = 0;
  for (double v : y) ss_parent += (v - mean_y) * (v - mean_y);
  std::cout << "SS(parent) " << ss_parent << "\n";

  double best_ss = 1e300;
  double best_t = 0;
  for (size_t i = 0; i + 1 < x.size(); ++i) {
    double t = (x[i] + x[i + 1]) / 2.0;
    std::vector<double> ly, ry;
    for (size_t r = 0; r < x.size(); ++r) (x[r] <= t ? ly : ry).push_back(y[r]);
    auto ss = [](const std::vector<double>& v) {
      if (v.empty()) return 0.0;
      double m = 0;
      for (double e : v) m += e;
      m /= v.size();
      double s = 0;
      for (double e : v) s += (e - m) * (e - m);
      return s;
    };
    double sse = ss(ly) + ss(ry);
    if (sse < best_ss) { best_ss = sse; best_t = t; }
  }

  std::vector<double> ly, ry;
  for (size_t r = 0; r < x.size(); ++r) (x[r] <= best_t ? ly : ry).push_back(y[r]);
  auto mean = [](const std::vector<double>& v) {
    double m = 0;
    for (double e : v) m += e;
    return m / v.size();
  };
  auto ss = [](const std::vector<double>& v) {
    double m = 0;
    for (double e : v) m += e;
    m /= v.size();
    double s = 0;
    for (double e : v) s += (e - m) * (e - m);
    return s;
  };
  std::cout << "best split threshold " << best_t << "  left mean "
            << mean(ly) << " (" << ly.size() << " rows)  right mean "
            << mean(ry) << " (" << ry.size() << " rows)\n";
  std::cout << "SS(left) " << ss(ly) << "  SS(right) " << ss(ry)
            << "  SS(children) " << best_ss << "\n";
  std::cout << "fraction of variance explained "
            << 1.0 - best_ss / ss_parent << "\n";
  std::cout << "(the tree would recurse into [5 4 5] and [8 7 11] next)\n";
  return 0;
}
