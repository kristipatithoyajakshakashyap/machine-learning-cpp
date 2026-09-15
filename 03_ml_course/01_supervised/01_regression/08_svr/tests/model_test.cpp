// 03_ml_course/01_supervised/01_regression/08_svr/tests/model_test.cpp
// Numerical fixtures for course::SVR. Every check() throws on failure so the
// process exits non-zero and CTest reports it.
//
// Reads/Writes: nothing.
// Run:  cmake --build --preset course --target svr_tests
//       ctest --preset course -R svr_numerical
#include "../Model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
// Minimal assertion: abort the whole test with a non-zero exit on the first failure.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Numerical fixture failed");
}
// Fixtures, in order (linear kernel, C = 100, epsilon = 0.01, 10000 iterations):
//   1. Exact line y = 2x + 1 is reproduced within 0.1 at every training point.
//   2. Dual feasibility: every |beta_i| <= C and sum(beta) == 0 (within 1e-8).
//   3. save() -> load() round trip predicts identically (tolerance 1e-12).
//   4. Empty input must be rejected with std::invalid_argument, not crash.
int main() {
  // 1. Linear SVR on an exact line: predictions within 0.1 of every target.
  course::SVR m(100, 0.01, false, 0.1, 10000);
  ml::Mat X = {{-2}, {-1}, {0}, {1}, {2}};
  ml::Vec y = {-3, -1, 1, 3, 5};
  m.fit(X, y);
  auto p = m.predict(X);
  for (size_t i = 0; i < y.size(); ++i)
    check(std::abs(p[i] - y[i]) < 0.1);
  // 2. Dual constraints: each |beta_i| <= C and the betas sum to zero.
  double sum = 0;
  for (double b : m.beta) {
    sum += b;
    check(std::abs(b) <= m.C + 1e-10);
  }
  check(std::abs(sum) < 1e-8);
  // 3. Text round trip: the reloaded model must predict bit-for-bit the same.
  std::stringstream stream;
  m.save(stream);
  course::SVR copy;
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
