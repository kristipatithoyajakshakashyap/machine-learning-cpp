// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/tests/model_test.cpp
//
// Purpose : numerical fixture test for ml::permutation_importance and
//           ml::learning_curve driven by the module's forest factory. A failed
//           check throws, so the CTest entry `diag_numerical` fails on a
//           non-zero exit code.
// Inputs  : none (a 120-row, 3-column fixture is generated here); no defines.
// Outputs : prints only (nothing is written to results/).
// Run     : target diag_tests; registered as `ctest -R diag_numerical`.
#include "../Forest.hpp"
#include "helper/eval/diagnostics.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
// Assertion helper: the message names the failing property in the exception.
void check(bool x, const char *what) {
  if (!x)
    throw std::runtime_error(std::string("Numerical fixture failed: ") + what);
}
int main() {
  // Feature 0 determines the class; feature 1 is constant; feature 2 is noise.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 1.0);
  ml::Mat X;
  ml::Vec y;
  for (int i = 0; i < 120; ++i) {
    const int c = i % 2;
    X.push_back({c * 4.0 + noise(rng) * 0.5, 1.0, noise(rng)});
    y.push_back(c);
  }
  ml::seed_rng(5);
  auto forest = course::make_forest(30);
  forest.fit(X, y);
  auto imp = ml::permutation_importance(forest, X, y, ml::accuracy, 10, 1);
  // One ImportanceRow per column, in column order.
  check(imp.size() == 3, "one row per feature");
  // Shuffling the only informative column must cost a lot of accuracy
  // (classes are 4 units apart with sd 0.5, so the forest relies on it).
  check(imp[0].mean_drop > 0.2, "true feature matters");
  // A constant column is never split on, so shuffling it changes nothing;
  // a noise column may be split on occasionally, hence the small tolerance.
  check(std::abs(imp[1].mean_drop) < 0.05, "constant feature is unimportant");
  check(std::abs(imp[2].mean_drop) < 0.05, "noise feature is unimportant");

  // Learning curve with 4 stratified folds at 10 %, 50 % and 100 % of the
  // training rows, scored by accuracy.
  auto curve = ml::learning_curve([] { return course::make_forest(20); }, X, y,
                                  {0.1, 0.5, 1.0}, 4, 2, ml::accuracy, true);
  // One LearningCurvePoint per fraction, in the order given.
  check(curve.size() == 3, "one point per fraction");
  // n_train must grow with the fraction (rounding cannot collapse two points).
  check(curve[0].n_train < curve[1].n_train && curve[1].n_train < curve[2].n_train,
        "training sizes increase");
  // On a separable fixture the CV score at 100 % cannot be below that at 10 %.
  check(curve[2].cv_score >= curve[0].cv_score, "more data does not hurt");

  // Input validation: an empty X/y must be rejected with invalid_argument
  // rather than crashing inside the shuffle loop.
  bool rejected = false;
  try {
    ml::permutation_importance(forest, {}, {}, ml::accuracy, 1, 0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "empty data throws");
}
