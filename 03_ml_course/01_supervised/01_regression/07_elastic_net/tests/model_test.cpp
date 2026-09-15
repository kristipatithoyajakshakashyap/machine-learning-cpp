// 03_ml_course/01_supervised/01_regression/07_elastic_net/tests/model_test.cpp
// Numerical fixtures for course::ElasticNet. Every check() throws on failure so
// the process exits non-zero and CTest reports it.
//
// Reads/Writes: nothing.
// Run:  cmake --build --preset course --target elastic_tests
//       ctest --preset course -R elastic_numerical
#include "../Model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
// Minimal assertion: abort the whole test with a non-zero exit on the first failure.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Numerical fixture failed");
}
// Fixtures, in order:
//   1. alpha = 0 reduces to least squares: y = 2x + 1 gives coef 2, intercept 1.
//   2. A huge lasso penalty (alpha 100, l1_ratio 1) zeroes the only coefficient.
//   3. save() -> load() round trip predicts identically (tolerance 1e-12).
//   4. Empty input must be rejected with std::invalid_argument, not crash.
int main() {
  // 1. No penalty: OLS on an exact line recovers slope 2 and intercept 1.
  course::ElasticNet m(0, 0);
  ml::Mat X = {{-2}, {-1}, {0}, {1}, {2}};
  ml::Vec y = {-3, -1, 1, 3, 5};
  m.fit(X, y);
  check(std::abs(m.coef[0] - 2) < 1e-8);
  check(std::abs(m.intercept - 1) < 1e-8);
  // 2. Pure lasso with a huge alpha shrinks the coefficient exactly to zero.
  course::ElasticNet penalized(100, 1);
  penalized.fit(X, y);
  check(penalized.coef[0] == 0);
  // 3. Text round trip: the reloaded model must predict bit-for-bit the same.
  std::stringstream stream;
  m.save(stream);
  course::ElasticNet copy;
  copy.load(stream);
  auto a = m.predict(X), b = copy.predict(X);
  for (size_t i = 0; i < a.size(); ++i)
    check(std::abs(a[i] - b[i]) < 1e-12);
  // 4. Input validation: empty X/y throws std::invalid_argument.
  bool rejected = false;
  try {
    m.fit({}, {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
}
