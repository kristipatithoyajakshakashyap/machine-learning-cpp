// 03_ml_course/01_supervised/01_regression/07_elastic_net/Model.hpp
// course::ElasticNet: linear regression with a mixed L1/L2 penalty, fitted by
// cyclic coordinate descent with soft-thresholding. Header-only; Model.cpp exists
// only so the ml_elastic library target has a translation unit.
//
// Reads/writes: nothing on disk; save()/load() serialise to a stream via
//               helper/persistence/archive.hpp (tag "ElasticNet_V1").
// Build:  library ml_elastic; used by elastic_theory, elastic_math_intuition,
//         elastic_implementation, elastic_end_to_end, elastic_predict and
//         elastic_tests (ctest -R elastic).
#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace course {
// Objective: ||y-Xw-b||^2/(2n) + alpha*l1_ratio*||w||_1
//            + alpha*(1-l1_ratio)*||w||^2/2. Intercept is unpenalized.
// Public fields are deliberately exposed so lessons can print and plot them:
//   alpha     overall penalty strength (0 = ordinary least squares),
//   l1_ratio  share of the penalty that is L1 (1 = lasso, 0 = ridge),
//   max_iter  cap on full coordinate sweeps,
//   coef      fitted weights w (empty until fit() succeeds),
//   intercept fitted bias b (never penalised).
class ElasticNet {
public:
  double alpha, l1_ratio;
  size_t max_iter;
  ml::Vec coef;
  double intercept = 0;
  // Parameters are validated in fit(), not here, so a default-constructed model
  // can be used as a load() target.
  ElasticNet(double a = 0.1, double ratio = 0.5, size_t iterations = 5000)
      : alpha(a), l1_ratio(ratio), max_iter(iterations) {}
  // Fit by coordinate descent on centred data.
  //   X: n rows of p finite features (all rows the same length); y: n finite targets.
  // Throws std::invalid_argument on empty/ragged/non-finite input or parameters
  // outside alpha >= 0, 0 <= l1_ratio <= 1, max_iter > 0.
  // Algorithm: centre X by column means and y by its mean (so the intercept drops
  // out of the penalty), then for each coordinate j compute
  //   rho_j = (1/n) x_j . (residual + x_j w_j)    (partial residual correlation)
  //   w_j   = soft(rho_j, alpha*l1_ratio) / (norm_j + alpha*(1-l1_ratio))
  // updating the cached residual after each change. Stop when the largest change
  // in one sweep is below 1e-9. Finally un-centre: b = mean(y) - mean(X) . w.
  void fit(const ml::Mat &X, const ml::Vec &y) {
    if (X.empty() || X[0].empty() || X.size() != y.size() || alpha < 0 ||
        l1_ratio < 0 || l1_ratio > 1 || !max_iter)
      throw std::invalid_argument("Invalid ElasticNet data/parameters");
    const size_t n = X.size(), p = X[0].size();
    for (size_t i = 0; i < n; ++i) {
      if (X[i].size() != p || !std::isfinite(y[i]))
        throw std::invalid_argument("Invalid row");
      for (double v : X[i])
        if (!std::isfinite(v))
          throw std::invalid_argument("Nonfinite feature");
    }
    auto center = ml::col_mean(X);
    intercept = 0;
    for (double v : y)
      intercept += v / n;
    coef.assign(p, 0);
    ml::Vec residual = y;
    for (double &v : residual)
      v -= intercept;
    ml::Vec norm(p, 0);
    for (const auto &row : X)
      for (size_t j = 0; j < p; ++j)
        norm[j] += (row[j] - center[j]) * (row[j] - center[j]) / n;
    // A zero-variance column has norm_j == 0 and alpha*(1-l1_ratio) may also be 0;
    // the `denom > 0` guard then leaves its weight at 0 instead of dividing by zero.
    for (size_t iter = 0; iter < max_iter; ++iter) {
      double change = 0;
      for (size_t j = 0; j < p; ++j) {
        double rho = 0;
        for (size_t i = 0; i < n; ++i)
          rho += (X[i][j] - center[j]) *
                 (residual[i] + (X[i][j] - center[j]) * coef[j]) / n;
        const double penalty = alpha * l1_ratio,
                     denom = norm[j] + alpha * (1 - l1_ratio);
        const double next =
            denom > 0
                ? std::copysign(std::max(0.0, std::abs(rho) - penalty), rho) /
                      denom
                : 0;
        const double delta = next - coef[j];
        change = std::max(change, std::abs(delta));
        for (size_t i = 0; i < n; ++i)
          residual[i] -= (X[i][j] - center[j]) * delta;
        coef[j] = next;
      }
      if (change < 1e-9)
        break;
    }
    for (size_t j = 0; j < p; ++j)
      intercept -= center[j] * coef[j];
  }
  // y_hat = X w + b for each row. Throws std::runtime_error before fit() and
  // std::invalid_argument when a row length differs from coef.size().
  ml::Vec predict(const ml::Mat &X) const {
    if (coef.empty())
      throw std::runtime_error("Model is not fitted");
    ml::Vec out;
    for (const auto &row : X) {
      if (row.size() != coef.size())
        throw std::invalid_argument("Feature count mismatch");
      out.push_back(ml::dot(row, coef) + intercept);
    }
    return out;
  }
  // Serialise the type tag followed by hyper-parameters and fitted values.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("ElasticNet_V1"));
    ml::archive::write(out, alpha, l1_ratio, max_iter, coef, intercept);
  }
  // Inverse of save(); rejects streams written by another model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("ElasticNet_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, alpha, l1_ratio, max_iter, coef, intercept);
  }
};
} // namespace course
