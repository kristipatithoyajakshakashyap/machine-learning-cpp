// 03_ml_course/tests/numerical/svm_guards.cpp
//
// Purpose : input-validation, fixture and persistence guards shared by
//           LinearSVM and KernelSVM. check_model runs the same contract
//           against both classes; check_archives builds deliberately
//           inconsistent archives and proves load() rejects them without
//           damaging the fitted model.
// Inputs  : none (a four-point 1-D fixture; no data files, no defines used).
// Outputs : prints only. Exit 0 on success, non-zero on the first failure.
// Target  : ml_svm_guards (CTest: ml_svm_guards)
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "01_supervised/02_classification/03_svm_classification/SVM.hpp"

namespace {
// Throw (failing the test process) when an assertion does not hold.
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Run `action` and require that it throws a std::exception.
template <class Action>
void rejects(Action action, const char* message) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

// Linearly separable 1-D fixture: negatives are class 0, positives class 1.
const ml::Mat X{{-2.0}, {-1.0}, {1.0}, {2.0}};
const ml::Vec y{0, 0, 1, 1};

// The behavioural contract every SVM class must satisfy. Model is taken by
// value so each call starts from an unfitted copy of the given configuration.
template <class Model>
void check_model(Model model) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // Training-data guards: predict before fit, empty/ragged/non-finite X,
  // label count mismatch, labels outside {0,1}, and single-class data.
  rejects([&] { model.predict(X); }, "unfitted prediction accepted");
  rejects([&] { model.fit({}, {}); }, "empty training accepted");
  rejects([&] { model.fit({{}, {}}, {0, 1}); }, "zero features accepted");
  rejects([&] { model.fit({{0}, {1, 2}}, {0, 1}); },
          "ragged training accepted");
  rejects([&] { model.fit({{0}, {nan}}, {0, 1}); }, "NaN feature accepted");
  rejects([&] { model.fit({{0}, {inf}}, {0, 1}); },
          "infinite feature accepted");
  rejects([&] { model.fit(X, {0, 1}); }, "label count mismatch accepted");
  for (double unsupported : {-1.0, 2.0, .6, nan, inf})
    rejects([&] { model.fit(X, {0, 0, 1, unsupported}); },
            "unsupported label silently remapped");
  rejects([&] { model.fit(X, {1, 1, 1, 1}); },
          "single-class training accepted");

  // Fixture: a separable problem must be classified exactly, with negative
  // margin scores on the class-0 side and positive on the class-1 side.
  model.fit(X, y);
  require(model.predict(X) == y, "separable fixture changed predictions");
  const ml::Vec scores = model.decision_function(X);
  require(scores[0] < 0 && scores[3] > 0, "decision signs incorrect");
  // Query guards: empty batch is fine; wrong width, ragged, NaN, inf are not.
  require(model.predict({}).empty(), "empty fitted batch should remain empty");
  rejects([&] { model.predict({{0, 1}}); },
          "extra prediction feature accepted");
  rejects([&] { model.decision_function({{}}); }, "missing feature accepted");
  rejects([&] { model.predict({{0}, {0, 1}}); }, "ragged query accepted");
  rejects([&] { model.decision_function({{nan}}); }, "NaN query accepted");
  rejects([&] { model.predict({{inf}}); }, "infinite query accepted");
  // Invalid refits fail before changing the previously fitted state.
  rejects([&] { model.fit(X, {0, 0, 1, 99}); }, "invalid refit accepted");
  require(model.decision_function(X) == scores, "invalid refit damaged model");

  // Archive round trip into a fresh object reproduces the decision scores;
  // an archive for a different model type is rejected without side effects.
  std::stringstream state;
  model.save(state);
  Model restored;
  restored.load(state);
  require(restored.decision_function(X) == scores,
          "archive round trip changed scores");
  std::stringstream truncated("\"WrongModel\"\n");
  rejects([&] { restored.load(truncated); }, "wrong archive model accepted");
  require(restored.decision_function(X) == scores,
          "rejected load damaged model");
}

// Model-specific archive defects. Each corrupt archive is written with
// ml::archive::write in the exact field order used by the model's save().
void check_archives() {
  // LinearSVM layout: name, C, learning rate, epochs, weights, bias.
  ml::LinearSVM linear(10, .01, 1000);
  linear.fit(X, y);
  const auto before_linear = linear.decision_function(X);
  std::stringstream empty_coefficients;
  ml::archive::write(empty_coefficients, std::string("LinearSVM"), 1.0, .01,
                     size_t(100), ml::Vec{}, 0.0);
  rejects([&] { linear.load(empty_coefficients); },
          "empty coefficient archive accepted");
  std::stringstream invalid_rate;
  ml::archive::write(invalid_rate, std::string("LinearSVM"), 1.0, -.01,
                     size_t(100), ml::Vec{1.0}, 0.0);
  rejects([&] { linear.load(invalid_rate); },
          "invalid archived parameters accepted");
  std::stringstream truncated("\"LinearSVM\" 1 0.01 100 2 1");
  rejects([&] { linear.load(truncated); }, "truncated coefficients accepted");
  require(linear.decision_function(X) == before_linear,
          "bad load damaged linear state");

  // KernelSVM layout: name, C, gamma, tolerance, passes, alpha, bias,
  // support rows, internal +/-1 labels, feature width. `damaged` writes one
  // archive with the given fields, requires rejection, and requires that the
  // fitted kernel model is untouched afterwards.
  ml::KernelSVM kernel(10, .5, 1e-3, 100);
  kernel.fit(X, y);
  const auto before_kernel = kernel.decision_function(X);
  const ml::Vec internal_labels{-1, -1, 1, 1};
  auto damaged = [&](const ml::Mat& features, const ml::Vec& labels,
                     const ml::Vec& alpha, size_t width, double gamma) {
    std::stringstream archive;
    ml::archive::write(archive, std::string("KernelSVM"), 10.0, gamma, 1e-3,
                       size_t(100), alpha, 0.0, features, labels, width);
    rejects([&] { kernel.load(archive); },
            "inconsistent kernel archive accepted");
    require(kernel.decision_function(X) == before_kernel,
            "bad load damaged kernel state");
  };
  damaged(X, internal_labels, {0, 0, 0}, 1, .5);  // Missing dual coefficient.
  damaged(X, {-1, 1}, {0, 0, 0, 0}, 1, .5);       // Missing internal labels.
  damaged(X, internal_labels, {0, 0, 0, 0}, 2, .5);  // Wrong declared width.
  damaged(X, internal_labels, {0, 0, 0, 0}, 0, .5);  // Zero declared width.
  damaged({{-2}, {-1, 0}, {1}, {2}}, internal_labels, {0, 0, 0, 0}, 1,
          .5);  // Ragged retained rows.
  damaged(X, {0, 0, 1, 1}, {0, 0, 0, 0}, 1,
          .5);  // Public labels in internal state.
  damaged(X, internal_labels, {0, 11, 0, 0}, 1,
          .5);  // Dual coefficient exceeds C.
  damaged(X, internal_labels, {0, 0, 0, 0}, 1,
          -1);  // Auto gamma is not a fitted state.
}
}  // namespace

int main() {
  // Constructor guards: C, learning rate, gamma and tolerance must be finite
  // and positive; epoch / pass counts must be at least 1.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double invalid : {0.0, -1.0, nan, inf}) {
    rejects([&] { ml::LinearSVM model(invalid); }, "invalid linear C accepted");
    rejects([&] { ml::LinearSVM model(1, invalid); },
            "invalid learning rate accepted");
    rejects([&] { ml::KernelSVM model(invalid); }, "invalid kernel C accepted");
    rejects([&] { ml::KernelSVM model(1, .5, invalid); },
            "invalid tolerance accepted");
  }
  for (double invalid : {0.0, -2.0, nan, inf})
    rejects([&] { ml::KernelSVM model(1, invalid); }, "invalid gamma accepted");
  rejects([] { ml::LinearSVM model(1, .01, 0); }, "zero epochs accepted");
  rejects([] { ml::KernelSVM model(1, .5, 1e-3, 0); }, "zero passes accepted");
  // Same contract for the linear model and both kernel configurations.
  check_model(ml::LinearSVM(10, .01, 1000));
  check_model(ml::KernelSVM(10, .5, 1e-3, 100));
  check_model(ml::KernelSVM(10));  // Valid automatic gamma remains supported.
  check_archives();
  std::cout
      << "SVM validation, fixture predictions, and archive guards passed\n";
}
