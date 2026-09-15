// 03_ml_course/01_supervised/01_regression/08_svr/Model.hpp
// course::SVR: epsilon-insensitive support vector regression (linear or RBF
// kernel) solved in the dual with FISTA (accelerated proximal gradient). The
// model keeps every training row as a potential support vector; beta[i] is the
// signed dual coefficient (alpha_i - alpha_i*) of row i. Header-only; Model.cpp
// exists only so the ml_svr library target has a translation unit.
//
// Reads/writes: nothing on disk; save()/load() serialise to a stream via
//               helper/persistence/archive.hpp (tag "SVR_V1").
// Build:  library ml_svr; used by svr_theory, svr_math_intuition,
//         svr_implementation, svr_end_to_end, svr_predict and svr_tests
//         (ctest -R svr).
#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace course {
// Public fields:
//   C         box constraint on |beta_i| (larger = less regularisation),
//   epsilon   half-width of the loss-free tube around the prediction,
//   gamma     RBF width: k(a, b) = exp(-gamma |a - b|^2),
//   rbf       true for the RBF kernel, false for the linear kernel a . b,
//   max_iter  cap on FISTA iterations,
//   support   copy of the training rows (empty until fitted),
//   beta      dual coefficients, one per training row, sum(beta) == 0,
//   intercept bias b recovered from the KKT conditions.
class SVR {
public:
  double C, epsilon, gamma;
  bool rbf;
  size_t max_iter;
  ml::Mat support;
  ml::Vec beta;
  double intercept = 0;
  // Parameters are validated in fit(), so a default-constructed model can be a
  // load() target.
  SVR(double c = 10, double eps = 0.1, bool radial = true, double g = 0.1,
      size_t iterations = 1500)
      : C(c), epsilon(eps), gamma(g), rbf(radial), max_iter(iterations) {}
  // Kernel value between two feature vectors of equal length.
  double kernel(const ml::Vec &a, const ml::Vec &b) const {
    if (!rbf)
      return ml::dot(a, b);
    double d = 0;
    for (size_t j = 0; j < a.size(); ++j)
      d += (a[j] - b[j]) * (a[j] - b[j]);
    return std::exp(-gamma * d);
  }
  // Solve the SVR dual for X (n >= 2 rows, p finite features) and finite targets y.
  // Throws std::invalid_argument on bad shapes or parameters (C > 0, epsilon >= 0,
  // gamma > 0, max_iter > 0). Steps:
  //   1. Build the n x n kernel matrix K and a Lipschitz bound (max row 1-norm)
  //      that gives a safe fixed step size 1 / bound.
  //   2. FISTA: gradient step on 0.5 beta.K.beta - y.beta, then a prox that
  //      soft-thresholds by step*epsilon and clips to [-C, C]; a bisection over a
  //      scalar shift (70 halvings) makes the prox output sum to exactly 0.
  //   3. Recover the intercept from free support vectors (0 < |beta| < C), or,
  //      if there are none, from the median of the epsilon-loss knots.
  void fit(const ml::Mat &X, const ml::Vec &y) {
    if (X.size() < 2 || X[0].empty() || X.size() != y.size() || C <= 0 ||
        epsilon < 0 || gamma <= 0 || !max_iter)
      throw std::invalid_argument("Invalid SVR data/parameters");
    const size_t n = X.size(), p = X[0].size();
    for (size_t i = 0; i < n; ++i) {
      if (X[i].size() != p || !std::isfinite(y[i]))
        throw std::invalid_argument("Invalid row");
      for (double v : X[i])
        if (!std::isfinite(v))
          throw std::invalid_argument("Nonfinite feature");
    }
    support = X;
    beta.assign(n, 0);
    ml::Mat K(n, ml::Vec(n));
    double bound = 0;
    for (size_t i = 0; i < n; ++i) {
      double sum = 0;
      for (size_t j = 0; j < n; ++j) {
        K[i][j] = kernel(X[i], X[j]);
        sum += std::abs(K[i][j]);
      }
      bound = std::max(bound, sum);
    }
    const double step = 1 / std::max(bound, 1e-12);
    ml::Vec accelerated = beta;
    double t = 1;
    // FISTA on the signed dual. The scalar multiplier makes the prox satisfy
    // sum(beta)=0 exactly; clipping enforces the dual box [-C,C].
    for (size_t iter = 0; iter < max_iter; ++iter) {
      ml::Vec z(n);
      for (size_t i = 0; i < n; ++i)
        z[i] = accelerated[i] - step * (ml::dot(K[i], accelerated) - y[i]);
      auto prox = [&](double v) {
        return std::clamp(
            std::copysign(std::max(0.0, std::abs(v) - step * epsilon), v), -C,
            C);
      };
      double lo = *std::min_element(z.begin(), z.end()) - C - step * epsilon;
      double hi = *std::max_element(z.begin(), z.end()) + C + step * epsilon;
      // Bisection: sum(prox(z - mid)) is monotone decreasing in mid, so the root is
      // bracketed by [lo, hi]; 70 halvings reduce the bracket far below double eps.
      for (int k = 0; k < 70; ++k) {
        double mid = (lo + hi) / 2, sum = 0;
        for (double v : z)
          sum += prox(v - mid);
        if (sum > 0)
          lo = mid;
        else
          hi = mid;
      }
      ml::Vec next(n);
      double change = 0;
      for (size_t i = 0; i < n; ++i) {
        next[i] = prox(z[i] - (lo + hi) / 2);
        change = std::max(change, std::abs(next[i] - beta[i]));
      }
      // Nesterov momentum schedule; the extrapolated point drives the next gradient step.
      // Convergence is declared only after 20 iterations so early tiny steps do not stop it.
      double next_t = (1 + std::sqrt(1 + 4 * t * t)) / 2;
      for (size_t i = 0; i < n; ++i)
        accelerated[i] = next[i] + (t - 1) / next_t * (next[i] - beta[i]);
      beta = std::move(next);
      t = next_t;
      if (change < 1e-8 && iter > 20)
        break;
    }
    ml::Vec free_bias, residual;
    for (size_t i = 0; i < n; ++i) {
      double r = y[i] - ml::dot(K[i], beta);
      residual.push_back(r);
      if (std::abs(beta[i]) > 1e-6 && std::abs(beta[i]) < C - 1e-6)
        free_bias.push_back(r - std::copysign(epsilon, beta[i]));
    }
    if (free_bias.empty()) {
      // For fixed dual coefficients the intercept minimizes the sum of
      // epsilon-insensitive residual losses. Its minimizer interval lies
      // between the middle two of the 2n residual +/- epsilon knots.
      ml::Vec knots;
      for (double r : residual) {
        knots.push_back(r - epsilon);
        knots.push_back(r + epsilon);
      }
      std::sort(knots.begin(), knots.end());
      intercept = (knots[n - 1] + knots[n]) / 2;
    } else {
      intercept = 0;
      for (double v : free_bias)
        intercept += v / free_bias.size();
    }
  }
  // f(x) = sum_i beta_i k(x, support_i) + b. Throws std::runtime_error before fit()
  // and std::invalid_argument on a feature-count mismatch.
  ml::Vec predict(const ml::Mat &X) const {
    if (support.empty())
      throw std::runtime_error("Model is not fitted");
    ml::Vec out;
    for (const auto &row : X) {
      if (row.size() != support[0].size())
        throw std::invalid_argument("Feature count mismatch");
      double v = intercept;
      for (size_t i = 0; i < support.size(); ++i)
        v += beta[i] * kernel(row, support[i]);
      out.push_back(v);
    }
    return out;
  }
  // Serialise the type tag, hyper-parameters, support rows, beta and intercept.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("SVR_V1"));
    ml::archive::write(out, C, epsilon, gamma, rbf, max_iter, support, beta,
                       intercept);
  }
  // Inverse of save(); rejects streams written by another model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("SVR_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, C, epsilon, gamma, rbf, max_iter, support, beta,
                      intercept);
  }
};
} // namespace course
