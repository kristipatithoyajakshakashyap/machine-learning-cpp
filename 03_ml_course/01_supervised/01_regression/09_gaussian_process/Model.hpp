// 03_ml_course/01_supervised/01_regression/09_gaussian_process/Model.hpp
// course::GaussianProcess: exact GP regression (header-only, O(n^3) fit).
//
// Reads/writes: nothing on disk; save()/load() serialise to a stream via
//               helper/persistence/archive.hpp (tag "GaussianProcess_V1").
// Build:  no separate library (ML_MODULE_LIB is ml_core); included directly by
//         gp_theory, gp_math_intuition, gp_implementation, gp_end_to_end,
//         gp_predict and gp_tests (ctest -R gp_).
#pragma once
// Gaussian process regression with an RBF kernel plus observation noise.
//   k(x, z) = signal_var * exp(-|x - z|^2 / (2 length_scale^2))
//   K = k(X, X) + noise_var I = L L^T (Cholesky), alpha = K^{-1} y
//   mean(x*) = k(x*, X) alpha
//   var(x*)  = k(x*, x*) - |L^{-1} k(X, x*)|^2      (latent f, noise-free)
//   log p(y | X) = -y^T alpha / 2 - sum log L_ii - n log(2 pi) / 2
// The pipeline parameter is the length scale; inputs are expected to be
// standardised and the target centred/scaled by the pipeline.
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace course {

// Used only in the log-marginal-likelihood normalising constant.
constexpr double kPi = 3.14159265358979323846;

// Hyper-parameters are fixed at construction (the pipeline searches over the
// length scale). State after fit(): X_ (training rows), L_ (Cholesky factor of
// K + noise I), alpha_ (= K^{-1} y) and lml_ (log marginal likelihood).
class GaussianProcess {
public:
  // Throws std::invalid_argument unless length_scale > 0, signal_var > 0 and
  // noise_var >= 0 (all finite).
  explicit GaussianProcess(double length_scale = 1.0, double signal_var = 1.0,
                           double noise_var = 0.1)
      : length_scale_(length_scale), signal_var_(signal_var),
        noise_var_(noise_var) {
    validate();
  }

  // Read-only hyper-parameter accessors.
  double length_scale() const { return length_scale_; }
  double signal_var() const { return signal_var_; }
  double noise_var() const { return noise_var_; }

  // Condition on X (n rows, equal length) and y (n values). Builds the symmetric
  // kernel matrix with noise on the diagonal, factorises it, solves for alpha and
  // caches the log marginal likelihood. Throws std::invalid_argument on empty or
  // ragged input and std::runtime_error if K is not positive definite.
  void fit(const ml::Mat &X, const ml::Vec &y) {
    if (X.empty() || X.size() != y.size() || X[0].empty())
      throw std::invalid_argument("GaussianProcess needs rows and labels");
    const size_t n = X.size();
    for (const auto &row : X)
      if (row.size() != X[0].size())
        throw std::invalid_argument("GaussianProcess: ragged rows");
    X_ = X;
    ml::Mat K(n, ml::Vec(n));
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j <= i; ++j)
        K[i][j] = K[j][i] = kernel(X[i], X[j]) + (i == j ? noise_var_ : 0.0);
    L_ = cholesky(K);
    // alpha = L^{-T} L^{-1} y
    alpha_ = back_solve(forward_solve(y));
    double quad = 0.0, logdet = 0.0;
    for (size_t i = 0; i < n; ++i) {
      quad += y[i] * alpha_[i];
      logdet += std::log(L_[i][i]);
    }
    lml_ = -0.5 * quad - logdet -
           0.5 * static_cast<double>(n) * std::log(2.0 * kPi);
  }

  // Posterior mean at each query row: k(x*, X) . alpha.
  ml::Vec predict(const ml::Mat &Xs) const {
    require_fitted(Xs);
    ml::Vec out(Xs.size());
    for (size_t s = 0; s < Xs.size(); ++s) {
      double m = 0.0;
      for (size_t i = 0; i < X_.size(); ++i)
        m += kernel(Xs[s], X_[i]) * alpha_[i];
      out[s] = m;
    }
    return out;
  }

  // Posterior standard deviation of the latent function (no noise term).
  ml::Vec predict_std(const ml::Mat &Xs) const {
    require_fitted(Xs);
    ml::Vec out(Xs.size());
    for (size_t s = 0; s < Xs.size(); ++s) {
      ml::Vec ks(X_.size());
      for (size_t i = 0; i < X_.size(); ++i)
        ks[i] = kernel(Xs[s], X_[i]);
      const ml::Vec v = forward_solve(ks);
      double var = signal_var_;
      for (double vi : v)
        var -= vi * vi;
      out[s] = std::sqrt(std::max(var, 0.0));
    }
    return out;
  }

  // log p(y | X) computed in fit(); throws std::runtime_error before fitting.
  double log_marginal_likelihood() const {
    if (X_.empty())
      throw std::runtime_error("GaussianProcess is not fitted");
    return lml_;
  }

  // Serialise the type tag, hyper-parameters and the fitted state (X_, alpha_, L_).
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("GaussianProcess_V1"), length_scale_,
                       signal_var_, noise_var_, lml_, X_, alpha_, L_);
  }
  // Inverse of save(); re-validates hyper-parameters and checks that the fitted
  // arrays agree in size so a truncated archive cannot produce a half-loaded model.
  void load(std::istream &in) {
    std::string tag;
    ml::archive::read(in, tag);
    if (tag != "GaussianProcess_V1")
      throw std::runtime_error("Wrong model type: " + tag);
    ml::archive::read(in, length_scale_, signal_var_, noise_var_, lml_, X_,
                      alpha_, L_);
    validate();
    if (X_.size() != alpha_.size() || L_.size() != X_.size())
      throw std::runtime_error("GaussianProcess archive is inconsistent");
  }

private:
  // Hyper-parameter domain checks shared by the constructor and load().
  void validate() const {
    if (!(length_scale_ > 0.0) || !std::isfinite(length_scale_))
      throw std::invalid_argument("length_scale must be positive");
    if (!(signal_var_ > 0.0) || !std::isfinite(signal_var_))
      throw std::invalid_argument("signal_var must be positive");
    if (!(noise_var_ >= 0.0) || !std::isfinite(noise_var_))
      throw std::invalid_argument("noise_var must be non-negative");
  }
  // Common guard for predict()/predict_std(): fitted, and query width matches.
  void require_fitted(const ml::Mat &Xs) const {
    if (X_.empty())
      throw std::runtime_error("GaussianProcess is not fitted");
    for (const auto &row : Xs)
      if (row.size() != X_[0].size())
        throw std::invalid_argument("GaussianProcess: feature count mismatch");
  }
  // Squared-exponential kernel: signal_var * exp(-|a - b|^2 / (2 l^2)).
  double kernel(const ml::Vec &a, const ml::Vec &b) const {
    double d2 = 0.0;
    for (size_t j = 0; j < a.size(); ++j)
      d2 += (a[j] - b[j]) * (a[j] - b[j]);
    return signal_var_ * std::exp(-0.5 * d2 / (length_scale_ * length_scale_));
  }
  // Lower-triangular L with L L^T = K (Cholesky-Banachiewicz, row by row).
  // A non-positive pivot means K is numerically singular; the error message
  // suggests the standard fix of adding more noise to the diagonal.
  static ml::Mat cholesky(const ml::Mat &K) {
    const size_t n = K.size();
    ml::Mat L(n, ml::Vec(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        double s = K[i][j];
        for (size_t k = 0; k < j; ++k)
          s -= L[i][k] * L[j][k];
        if (i == j) {
          if (!(s > 0.0))
            throw std::runtime_error(
                "GaussianProcess: kernel matrix is not positive definite; "
                "increase noise_var");
          L[i][i] = std::sqrt(s);
        } else
          L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  // Solve L v = b.
  ml::Vec forward_solve(const ml::Vec &b) const {
    const size_t n = L_.size();
    ml::Vec v(n);
    for (size_t i = 0; i < n; ++i) {
      double s = b[i];
      for (size_t k = 0; k < i; ++k)
        s -= L_[i][k] * v[k];
      v[i] = s / L_[i][i];
    }
    return v;
  }
  // Solve L^T w = v.
  ml::Vec back_solve(const ml::Vec &v) const {
    const size_t n = L_.size();
    ml::Vec w(n);
    for (size_t ii = n; ii-- > 0;) {
      double s = v[ii];
      for (size_t k = ii + 1; k < n; ++k)
        s -= L_[k][ii] * w[k];
      w[ii] = s / L_[ii][ii];
    }
    return w;
  }

  double length_scale_, signal_var_, noise_var_;
  double lml_ = 0.0;
  ml::Mat X_, L_;
  ml::Vec alpha_;
};

} // namespace course
