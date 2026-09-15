// 03_ml_course/tests/numerical/nb_guards.cpp
//
// Purpose : input-validation, numerical and persistence guards for
//           GaussianNB. Proves that bad constructor arguments, bad training
//           data and corrupt archives are rejected, that constant features
//           fall back to the class priors, and that save/load is lossless.
// Inputs  : none (tiny in-memory matrices; no data files, no defines used).
// Outputs : prints only. Exit 0 on success, non-zero on the first failure.
// Target  : ml_nb_guards (CTest: ml_nb_guards)
#include "01_supervised/02_classification/04_naive_bayes/GaussianNB.hpp"
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
// Throw (and thus fail the test process) when an assertion does not hold.
void check(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
}
// Run `operation` and require that it throws a std::exception.
// Used for every "invalid input must be rejected" assertion below.
template<class Function> void rejects(Function operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "GaussianNB accepted invalid input");
}
}

int main() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  // Constructor: the variance-smoothing factor must be finite and >= 0.
  rejects([] { ml::GaussianNB invalid(-1); });
  rejects([&] { ml::GaussianNB invalid(nan); });
  rejects([&] { ml::GaussianNB invalid(infinity); });
  ml::GaussianNB model;
  rejects([&] { model.predict({{1}}); });             // predict before fit
  rejects([&] { model.fit({}, {}); });                // no rows
  rejects([&] { model.fit({{}, {}}, {0, 1}); });      // rows with no columns
  rejects([&] { model.fit({{1}, {2, 3}}, {0, 1}); }); // ragged rows
  rejects([&] { model.fit({{1}, {nan}}, {0, 1}); });  // NaN feature
  rejects([&] { model.fit({{1}, {infinity}}, {0, 1}); }); // inf feature
  // Labels must be contiguous class codes 0..K-1: negative, fractional,
  // NaN, inf, huge and gapped (0,2) encodings are all rejected.
  for (const auto& labels : ml::Mat{{-1, 0}, {0, 0.5}, {0, nan}, {0, infinity}, {0, 1e30}, {0, 2}})
    rejects([&] { model.fit({{1}, {2}}, labels); });
  // Class 1 is never observed although class 2 is: gap in the encoding.
  rejects([&] { model.fit({{1}, {2}, {3}}, {0, 2, 2}); });

  // Equal likelihoods reduce to empirical priors even when every feature is
  // constant and relative smoothing by itself would vanish.
  // Priors are 3/4 and 1/4; the variance floor keeps the Gaussian defined.
  model.fit({{7, 7}, {7, 7}, {7, 7}, {7, 7}}, {0, 0, 0, 1});
  const auto p = model.predict_proba({{7, 7}}).front();
  check(std::abs(p[0] - 0.75) < 1e-12 && std::abs(p[1] - 0.25) < 1e-12,
        "Constant features did not preserve priors");
  check(model.class_var()[0][0] > 0, "Variance floor absent");
  // Query-time guards: wrong width (1 or 3 columns instead of 2), NaN, inf.
  rejects([&] { model.predict({{7}}); });
  rejects([&] { model.predict_proba({{7, 7, 7}}); });
  rejects([&] { model.predict({{nan, 7}}); });
  rejects([&] { model.predict_proba({{7, infinity}}); });
  // An empty batch is valid and yields an empty result.
  check(model.predict({}).empty(), "Empty prediction batch should be supported");
  // Archive round trip into a fresh object must reproduce probabilities.
  std::stringstream saved;
  model.save(saved);
  ml::GaussianNB restored;
  restored.load(saved);
  check(restored.predict_proba({{7, 7}}) == model.predict_proba({{7, 7}}),
        "Archive round trip changed probabilities");
  // Five hand-built archives, each with exactly one defect:
  //   0 - one prior for two classes      1 - ragged mean matrix
  //   2 - zero variance                  3 - non-contiguous label 2
  //   4 - priors that do not sum to 1
  // Each must be rejected AND must leave `restored` fitted as before.
  for (int defect = 0; defect < 5; ++defect) {
    std::stringstream corrupt;
    ml::Vec priors = defect == 0 ? ml::Vec{1} : ml::Vec{0.5, 0.5};
    ml::Mat means = defect == 1 ? ml::Mat{{0}, {0, 1}} : ml::Mat{{0}, {1}};
    ml::Mat variances = defect == 2 ? ml::Mat{{0}, {1}} : ml::Mat{{1}, {1}};
    ml::Vec labels = defect == 3 ? ml::Vec{0, 2} : ml::Vec{0, 1};
    if (defect == 4) priors = {0.2, 0.2};
    // Layout mirrors GaussianNB::save: name, smoothing, n_features, priors,
    // means, variances, class labels.
    ml::archive::write(corrupt, std::string("GaussianNB"), 1e-9, size_t(2),
                       priors, means, variances, labels);
    rejects([&] { restored.load(corrupt); });
    check(restored.predict_proba({{7, 7}}) == model.predict_proba({{7, 7}}),
          "Rejected archive damaged existing fitted state");
  }
  // Edge case: smoothing 0 with a single class and zero variance must still
  // produce a well-defined probability of exactly 1 (no division by zero).
  ml::GaussianNB zero_smoothing(0);
  zero_smoothing.fit({{2}, {2}}, {0, 0});
  check(zero_smoothing.predict_proba({{2}})[0][0] == 1,
        "Single-class zero-smoothing constant dataset failed");
}
