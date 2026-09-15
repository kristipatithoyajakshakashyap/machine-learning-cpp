// 03_ml_course/01_supervised/02_classification/11_lda/tests/model_test.cpp
//
// Purpose : numerical fixture test for course::LDA. Any failed check throws, so a
//           non-zero exit code marks the CTest entry `lda_numerical` as failed.
// Inputs  : none (hand-written 1-D fixture); no defines used.
// Outputs : prints only (nothing is written to results/).
// Run     : target lda_tests; registered as `ctest -R lda_numerical`.
#include "../Model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
// Minimal assertion helper: throw instead of returning so main() aborts with a
// non-zero status on the first failure.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Numerical fixture failed");
}
int main() {
  course::LDA m;
  // Two classes separated at zero; labels 7 and 9 are deliberately not 0/1 to
  // prove that arbitrary label values round-trip through fit()/predict().
  ml::Mat X = {{-3}, {-2}, {-1}, {1}, {2}, {3}};
  ml::Vec y = {7, 7, 7, 9, 9, 9};
  m.fit(X, y);
  // Separable fixture: every training row must be classified correctly and
  // predict() must return the original label values, not class positions.
  check(m.predict(X) == y);
  // Posterior rows are normalised by log-sum-exp, so they must sum to one.
  for (const auto &row : m.predict_proba(X))
    check(std::abs(row[0] + row[1] - 1) < 1e-12);
  // Persistence round trip: a model loaded from save() output must reproduce
  // the same predictions (labels, priors, means and factors all survive).
  std::stringstream stream;
  m.save(stream);
  course::LDA copy;
  copy.load(stream);
  auto a = m.predict(X), b = copy.predict(X);
  for (size_t i = 0; i < a.size(); ++i)
    check(std::abs(a[i] - b[i]) < 1e-12);
  // Input validation: empty data must be rejected with std::invalid_argument
  // rather than crashing or silently producing an unfitted model.
  bool rejected = false;
  try {
    m.fit({}, {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
}
