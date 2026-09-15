// 03_ml_course/01_supervised/03_model_diagnostics/03_hyperparameter_search/tests/model_test.cpp
//
// Purpose : numerical fixture test for the search helpers (grid_search,
//           random_search, nested_cv) using a learner whose CV score is known
//           in closed form, plus a save/load check of course::TunedForest.
//           A failed check throws, so the CTest entry `hps_numerical` fails
//           on a non-zero exit code.
// Inputs  : none (fixtures are generated here); no defines used.
// Outputs : prints only (nothing is written to results/).
// Run     : target hps_tests; registered as `ctest -R hps_numerical`.
#include "../Model.hpp"
#include "helper/eval/search.hpp"
#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
// Assertion helper: the message names the failing property in the exception.
void check(bool x, const char *what) {
  if (!x)
    throw std::runtime_error(std::string("Numerical fixture failed: ") + what);
}
// Deterministic toy learner: predicts the constant `c`; CV score is exactly
// -(c - mean)^2, so the brute-force optimum is known in closed form.
struct Constant {
  double c = 0;
  void fit(const ml::Mat &, const ml::Vec &) {}
  ml::Vec predict(const ml::Mat &X) const { return ml::Vec(X.size(), c); }
};
// Higher-is-better wrapper around MSE for the search helpers.
double neg_mse(const ml::Vec &y, const ml::Vec &p) { return -ml::mse(y, p); }
int main() {
  // 20 rows with targets 0,1,2,3,4 repeated; the MSE-optimal constant is 2.
  ml::Mat X(20, ml::Vec(1, 0.0));
  ml::Vec y(20);
  for (size_t i = 0; i < 20; ++i)
    y[i] = double(i % 5);
  // 5 x 2 = 10 combinations; "unused" must not change the score.
  ml::ParamSpace space{{{"c", {0, 1, 2, 3, 4}}, {"unused", {0, 1}}}};
  auto make = [](const ml::Params &p) {
    Constant m;
    m.c = ml::param_value(p, "c");
    return m;
  };
  // grid_search best == brute force max
  // Every combination is visited exactly once, best_row() returns the row
  // with the largest mean, and that row has c = 2 (the closed-form optimum).
  auto grid = ml::grid_search(make, X, y, space, 4, 1, neg_mse, false);
  check(grid.size() == 10, "grid visits every combination");
  double brute = -1e300;
  for (const auto &r : grid)
    brute = std::max(brute, r.mean);
  check(std::abs(ml::best_row(grid).mean - brute) < 1e-15, "best equals brute-force max");
  check(ml::param_value(ml::best_row(grid).params, "c") == 2.0, "c=2 minimises MSE of 0..4");
  // random_search draws n_iter distinct combinations
  // Sampling is without replacement, so 6 draws give 6 distinct parameter
  // strings, and asking for 50 draws is capped at the 10 cells that exist.
  auto random = ml::random_search(make, X, y, space, 6, 4, 3, neg_mse, false);
  check(random.size() == 6, "n_iter rows");
  std::set<std::string> seen;
  for (const auto &r : random)
    seen.insert(ml::params_to_string(r.params));
  check(seen.size() == 6, "draws are distinct");
  auto capped = ml::random_search(make, X, y, space, 50, 4, 3, neg_mse, false);
  check(capped.size() == 10, "n_iter is capped at the space size");
  // nested_cv outer count
  // One NestedFoldResult per outer fold (5), and the naive score (plain best
  // CV on all rows) must be a finite number.
  auto nested = ml::nested_cv(make, X, y, space, 5, 3, 4, 9, neg_mse, false);
  check(nested.folds.size() == 5, "one outer score per outer fold");
  check(std::isfinite(nested.naive_score), "naive score finite");
  // TunedForest save/load exact
  // Two well-separated classes in column 0; column 1 is noise.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 1.0);
  ml::Mat Xf;
  ml::Vec yf;
  for (int i = 0; i < 80; ++i) {
    const int c = i % 2;
    Xf.push_back({c * 4.0 + noise(rng), noise(rng)});
    yf.push_back(c);
  }
  ml::seed_rng(5);
  course::TunedForest forest(7);
  // Index 7 -> depth kDepths[7 % 3 = 1] = 8, leaf kLeaves[7 / 3 = 2] = 5.
  check(forest.max_depth() == 8 && forest.min_leaf() == 5, "index 7 -> depth 8, leaf 5");
  forest.fit(Xf, yf);
  std::stringstream stream;
  forest.save(stream);
  course::TunedForest copy;
  copy.load(stream);
  // The archive restores the grid index and the trees bit for bit, so labels
  // and probabilities are identical, not merely close.
  check(copy.index() == 7, "index restored");
  auto a = forest.predict(Xf), b = copy.predict(Xf);
  auto pa = forest.predict_proba(Xf), pb = copy.predict_proba(Xf);
  for (size_t i = 0; i < a.size(); ++i) {
    check(a[i] == b[i], "reloaded predictions identical");
    for (size_t j = 0; j < pa[i].size(); ++j)
      check(pa[i][j] == pb[i][j], "reloaded probabilities identical");
  }
  // Input validation: index 9 is one past the 3 x 3 grid and must be refused
  // at construction with std::invalid_argument.
  bool rejected = false;
  try {
    course::TunedForest bad(9);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "index 9 rejected");
}
