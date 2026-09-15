// 03_ml_course/01_supervised/02_classification/12_qda/tests/model_test.cpp
//
// Purpose : numerical fixture test for course::QDA (LDA with per-class
//           covariance). Any failed check throws, so the CTest entry
//           `qda_numerical` fails on a non-zero exit code.
// Inputs  : none (hand-written 1-D fixture); no defines used.
// Outputs : prints only (nothing is written to results/).
// Run     : target qda_tests; registered as `ctest -R qda_numerical`.
#include "../Model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
// Minimal assertion helper: throwing makes main() exit non-zero at once.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Numerical fixture failed");
}
int main() {
  course::QDA m;
  // Two classes on either side of zero with non-consecutive labels 7 and 9 so
  // the label mapping (not just class positions) is exercised.
  ml::Mat X = {{-3}, {-2}, {-1}, {1}, {2}, {3}};
  ml::Vec y = {7, 7, 7, 9, 9, 9};
  m.fit(X, y);
  // Separable fixture: every row correct and original label values returned.
  check(m.predict(X) == y);
  // Posteriors are normalised with log-sum-exp and must sum to one per row.
  for (const auto &row : m.predict_proba(X))
    check(std::abs(row[0] + row[1] - 1) < 1e-12);
  // Round trip through save()/load(): the "QDA_V1" tag is accepted and every
  // per-class factor/log-determinant survives, so predictions are identical.
  std::stringstream stream;
  m.save(stream);
  course::QDA copy;
  copy.load(stream);
  auto a = m.predict(X), b = copy.predict(X);
  for (size_t i = 0; i < a.size(); ++i)
    check(std::abs(a[i] - b[i]) < 1e-12);
  // Input validation inherited from LDA: empty data raises invalid_argument.
  bool rejected = false;
  try {
    m.fit({}, {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
}
