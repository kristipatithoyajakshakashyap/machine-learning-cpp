// 03_ml_course/01_supervised/01_regression/09_gaussian_process/tests/model_test.cpp
// Numerical fixtures for course::GaussianProcess. Every check() throws on
// failure so the process exits non-zero and CTest reports it.
//
// Reads/Writes: nothing.
// Run:  cmake --build --preset course --target gp_tests
//       ctest --preset course -R gp_numerical
#include "../Model.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
namespace {
// Minimal assertion: abort the whole test with a non-zero exit on the first failure.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Numerical fixture failed");
}
// n equally spaced 1-D inputs from lo to hi inclusive (n >= 2).
ml::Mat grid(double lo, double hi, size_t n) {
  ml::Mat X;
  for (size_t i = 0; i < n; ++i)
    X.push_back({lo + (hi - lo) * static_cast<double>(i) /
                          static_cast<double>(n - 1)});
  return X;
}
} // namespace
// Fixtures: interpolation with near-zero noise, uncertainty growth away from
// the data, exact save/load round trip, and rejection of bad hyper-parameters
// and of predict() before fit().
int main() {
  // Arrange: smooth 1-D target, tiny noise so the posterior interpolates.
  const ml::Mat X = grid(0.0, 3.0, 12);
  ml::Vec y(X.size());
  for (size_t i = 0; i < X.size(); ++i)
    y[i] = std::sin(X[i][0]);
  course::GaussianProcess gp(1.0, 1.0, 1e-8);
  gp.fit(X, y);

  // Interpolation: mean at the training inputs reproduces the targets.
  const ml::Vec mean = gp.predict(X);
  for (size_t i = 0; i < y.size(); ++i)
    check(std::fabs(mean[i] - y[i]) < 1e-4);
  check(std::isfinite(gp.log_marginal_likelihood()));

  // Uncertainty grows away from the data.
  // At the training inputs the posterior std is ~sqrt(noise) (tiny); at x = 10, far
  // from every observation, it must return to almost the prior std (sqrt(1) = 1).
  const ml::Vec sd_train = gp.predict_std(X);
  const ml::Vec sd_far = gp.predict_std({{10.0}});
  for (double s : sd_train)
    check(s < 1e-2);
  check(sd_far[0] > 0.9 && sd_far[0] > sd_train[0]);

  // Save / load round trip is exact.
  std::stringstream stream;
  gp.save(stream);
  course::GaussianProcess copy;
  copy.load(stream);
  const ml::Mat Xq = grid(-1.0, 4.0, 7);
  const ml::Vec a = gp.predict(Xq), b = copy.predict(Xq);
  const ml::Vec sa = gp.predict_std(Xq), sb = copy.predict_std(Xq);
  for (size_t i = 0; i < a.size(); ++i) {
    // Exact equality is intended: the archive stores every double bit for bit.
    check(a[i] == b[i]);
    check(sa[i] == sb[i]);
  }
  check(copy.log_marginal_likelihood() == gp.log_marginal_likelihood());

  // Invalid hyperparameters and unfitted use are rejected.
  bool rejected = false;
  try {
    // length_scale 0 and -1 violate the > 0 domain; predicting on an unfitted model
    // is a runtime_error, not an invalid_argument.
    course::GaussianProcess bad(0.0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
  rejected = false;
  try {
    course::GaussianProcess neg(-1.0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
  rejected = false;
  try {
    course::GaussianProcess fresh;
    fresh.predict(X);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected);
  return 0;
}
