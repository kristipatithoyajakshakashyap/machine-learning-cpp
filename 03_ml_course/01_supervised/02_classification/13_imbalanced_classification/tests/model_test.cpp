// 03_ml_course/01_supervised/02_classification/13_imbalanced_classification/tests/model_test.cpp
//
// Purpose : numerical fixture test for course::WeightedLogistic. A failed check
//           throws, so the CTest entry `cimb_numerical` fails on a non-zero exit.
// Inputs  : none (two hand-written 2-D fixtures); no defines used.
// Outputs : prints only (nothing is written to results/).
// Run     : target cimb_tests; registered as `ctest -R cimb_numerical`.
#include "../Model.hpp"
#include "helper/math/metrics.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
// Assertion helper: the message names the failing property in the exception.
void check(bool x, const char *what) {
  if (!x)
    throw std::runtime_error(std::string("Numerical fixture failed: ") + what);
}
int main() {
  // Separable 2-D fixture: weight 1 classifies every row correctly.
  ml::Mat X = {{-2, -1}, {-1.5, -2}, {-1, -1.2}, {1, 1.5}, {1.5, 1}, {2, 2}};
  ml::Vec y = {0, 0, 0, 1, 1, 1};
  course::WeightedLogistic m(1.0);
  m.fit(X, y);
  // Plain (unweighted) logistic regression must separate the two clusters.
  check(m.predict(X) == y, "separable fit");
  // predict_proba returns {1 - p, p}, so the two entries always sum to one.
  for (const auto &row : m.predict_proba(X))
    check(std::abs(row[0] + row[1] - 1) < 1e-12, "probabilities sum to one");

  // Imbalanced overlapping fixture: heavier positive weight cannot lower
  // recall.
  // 40 negatives on a line from x = -1 to 0.95 with a small y wobble, and only
  // 4 positives starting at x = 0.4, so the two classes overlap in x.
  ml::Mat Xi;
  ml::Vec yi;
  for (int i = 0; i < 40; ++i) {
    Xi.push_back({-1.0 + 0.05 * i, 0.3 * ((i % 3) - 1)});
    yi.push_back(0);
  }
  for (int i = 0; i < 4; ++i) {
    Xi.push_back({0.4 + 0.3 * i, 0.2 * i});
    yi.push_back(1);
  }
  // Same learning rate and epoch budget; only the positive weight differs.
  course::WeightedLogistic light(1.0, 0.1, 500), heavy(8.0, 0.1, 500);
  light.fit(Xi, yi);
  heavy.fit(Xi, yi);
  double r1 = ml::binary_scores(yi, light.predict(Xi)).recall;
  double r8 = ml::binary_scores(yi, heavy.predict(Xi)).recall;
  // The whole point of cost-sensitive learning: w = 8 shifts the boundary
  // toward the negatives, so recall on positives can only stay or rise.
  check(r8 >= r1, "recall grows with weight");

  // Persistence round trip reproduces probabilities.
  std::stringstream stream;
  heavy.save(stream);
  course::WeightedLogistic copy;
  copy.load(stream);
  auto a = heavy.predict_proba(Xi), b = copy.predict_proba(Xi);
  for (size_t i = 0; i < a.size(); ++i)
    check(std::abs(a[i][1] - b[i][1]) < 1e-12, "reload probabilities");
  // Hyper-parameters travel with the archive (the copy was built with w = 1).
  check(copy.positive_weight() == 8.0, "reload hyperparameters");

  // Input validation: empty X/y must raise std::invalid_argument.
  bool rejected = false;
  try {
    m.fit({}, {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "empty fit throws");
}
