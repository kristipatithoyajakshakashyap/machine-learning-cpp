// 03_ml_course/02_unsupervised/05_gaussian_mixture/GaussianMixture.hpp
// Purpose: self-contained diagonal-covariance Gaussian mixture model fitted by
//          Expectation-Maximisation (EM); used by the ugm_* lessons, ugm_predict
//          and tests/verify_reload.cpp. Header-only; no .cpp pair.
// Inputs:  a dense finite feature matrix passed to fit(); hyper-parameters
//          k (components), iterations, seed and reg (variance floor).
// Outputs: none directly (no printing, no files); callers use predict_proba(),
//          predict(), score(), bic(), aic() and persist with save()/load().
// Run target: not an executable; linked via the ml_gmm INTERFACE library.
//
// Model: p(x) = sum_c w_c * N(x | mu_c, diag(var_c)). "Diagonal" means every
// component is an axis-aligned Gaussian (no feature correlations), which keeps
// the maths to per-feature means/variances and makes EM cheap and stable.
// EM alternates: E-step  r_ic = p(component c | x_i)   (responsibilities)
//                M-step  w_c, mu_c, var_c <- responsibility-weighted stats.
// Every EM step is guaranteed not to decrease the log-likelihood.
// sklearn equivalent: sklearn.mixture.GaussianMixture(n_components=k,
//   covariance_type='diag', reg_covar=reg, max_iter=iterations,
//   random_state=seed, init_params='random').
#pragma once
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// Diagonal covariance mixture: variances are independently regularized.
class GaussianMixture {
  size_t k_, iterations_;       // components; EM iteration budget
  unsigned seed_;               // seed for the random initial means
  double reg_, log_likelihood_ = 0;  // variance floor; final training LL
  Mat means_, variances_;       // k x p each; variances_ > 0 after fit
  Vec weights_;                 // k mixing weights, sum to 1
  // Un-normalised log posterior of each component for one row:
  //   log w_c + sum_j [ -1/2 log(2 pi var_cj) - (x_j - mu_cj)^2 / (2 var_cj) ].
  // Working in log space avoids underflow when densities are tiny. O(k p).
  Vec log_terms(const Vec& x) const {
    Vec z(k_);
    for (size_t c = 0; c < k_; ++c) {
      z[c] = std::log(weights_[c]);
      for (size_t j = 0; j < x.size(); ++j)
        z[c] -= .5 * (std::log(6.283185307179586 * variances_[c][j]) +
                      (x[j] - means_[c][j]) * (x[j] - means_[c][j]) /
                          variances_[c][j]);
    }
    return z;
  }

 public:
  // k: number of components; iterations: max EM steps; seed: chooses the k
  // rows used as initial means; reg: added to every variance so no component
  // can collapse onto a single point (variance -> 0, likelihood -> infinity).
  GaussianMixture(size_t k = 3, size_t iterations = 200, unsigned seed = 42,
                  double reg = 1e-6)
      : k_(k), iterations_(iterations), seed_(seed), reg_(reg) {}
  // Run EM on X. Deterministic for a fixed seed. Stops early once the
  // log-likelihood improves by less than 1e-7 per row. Complexity per
  // iteration O(n k p). Throws on invalid data or parameters.
  void fit(const Mat& X) {
    validate_dense(X);
    if (!k_ || k_ > X.size() || !iterations_ || !(reg_ > 0))
      throw std::invalid_argument("invalid mixture parameters");
    size_t p = X[0].size();
    // Initialisation: means = k distinct rows chosen by a seeded shuffle,
    // unit variances, equal weights.
    std::vector<size_t> ids(X.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed_);
    std::shuffle(ids.begin(), ids.end(), rng);
    means_.clear();
    for (size_t c = 0; c < k_; ++c) means_.push_back(X[ids[c]]);
    variances_ = Mat(k_, Vec(p, 1));
    weights_ = Vec(k_, 1.0 / k_);
    double last = -std::numeric_limits<double>::infinity();
    for (size_t it = 0; it < iterations_; ++it) {
      // E-step: responsibilities R[i][c] under the current parameters.
      Mat R = predict_proba(X);
      // M-step, part 1: effective counts nk and weighted feature sums.
      Vec nk(k_, 0);
      Mat mu(k_, Vec(p, 0)), var(k_, Vec(p, 0));
      for (size_t i = 0; i < X.size(); ++i)
        for (size_t c = 0; c < k_; ++c) {
          nk[c] += R[i][c];
          for (size_t j = 0; j < p; ++j) mu[c][j] += R[i][c] * X[i][j];
        }
      for (size_t c = 0; c < k_; ++c)
        for (size_t j = 0; j < p; ++j) mu[c][j] /= std::max(nk[c], 1e-15);
      // M-step, part 2: weighted squared deviations around the new means.
      for (size_t i = 0; i < X.size(); ++i)
        for (size_t c = 0; c < k_; ++c)
          for (size_t j = 0; j < p; ++j)
            var[c][j] += R[i][c] * (X[i][j] - mu[c][j]) * (X[i][j] - mu[c][j]);
      for (size_t c = 0; c < k_; ++c) {
        if (nk[c] < 1e-10) {
          // Empty component: reseed it at its initial row rather than let
          // its variance become 0/0.
          mu[c] = X[ids[c]];
          var[c] = Vec(p, 1);
          nk[c] = 1e-10;
        } else
          for (size_t j = 0; j < p; ++j) var[c][j] = var[c][j] / nk[c] + reg_;
        weights_[c] = nk[c];
      }
      // Weights are the normalised effective counts.
      double total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
      for (double& w : weights_) w /= total;
      means_ = mu;
      variances_ = var;
      // Convergence check on the total log-likelihood (scaled by n).
      log_likelihood_ = score(X);
      if (std::abs(log_likelihood_ - last) < 1e-7 * X.size()) break;
      last = log_likelihood_;
    }
  }
  // Responsibilities: one row per input, k entries summing to 1. Uses the
  // log-sum-exp trick (subtract the max) so exp() never overflows/underflows.
  // Throws logic_error before fit(), invalid_argument on a width mismatch.
  Mat predict_proba(const Mat& X) const {
    if (means_.empty()) {
      throw std::logic_error("fit first");
    }
    validate_dense(X, means_[0].size());
    Mat R;
    for (const auto& x : X) {
      Vec z = log_terms(x);
      double m = *std::max_element(z.begin(), z.end()), sum = 0;
      for (double& v : z) {
        v = std::exp(v - m);
        sum += v;
      }
      for (double& v : z) v /= sum;
      R.push_back(z);
    }
    return R;
  }
  // Hard assignment: index of the most responsible component per row.
  Vec predict(const Mat& X) const {
    Vec y;
    for (const auto& r : predict_proba(X))
      y.push_back(double(std::max_element(r.begin(), r.end()) - r.begin()));
    return y;
  }
  // Total log-likelihood sum_i log p(x_i), again via log-sum-exp per row.
  double score(const Mat& X) const {
    if (means_.empty()) {
      throw std::logic_error("fit first");
    }
    validate_dense(X, means_[0].size());
    double ll = 0;
    for (const auto& x : X) {
      auto z = log_terms(x);
      double m = *std::max_element(z.begin(), z.end()), s = 0;
      for (double v : z) s += std::exp(v - m);
      ll += m + std::log(s);
    }
    return ll;
  }
  // Bayesian information criterion: -2 LL + params * log n. Lower is better.
  // Free parameters: k means x p, k variances x p, and k - 1 weights.
  double bic(const Mat& X) const {
    return -2 * score(X) +
           (2 * k_ * X[0].size() + k_ - 1) * std::log(double(X.size()));
  }
  // Akaike information criterion: -2 LL + 2 * params. Penalises size less
  // than BIC, so it tends to prefer more components.
  double aic(const Mat& X) const {
    return -2 * score(X) + 2 * double(2 * k_ * X[0].size() + k_ - 1);
  }
  // Persist hyper-parameters and the fitted means/variances/weights.
  void save(std::ostream& o) const {
    archive::write(o, k_, iterations_, seed_, reg_, log_likelihood_, means_,
                   variances_, weights_);
  }
  // Inverse of save(); checks all three parameter blocks have k entries.
  void load(std::istream& i) {
    archive::read(i, k_, iterations_, seed_, reg_, log_likelihood_, means_,
                  variances_, weights_);
    validate_dense(means_);
    if (means_.size() != k_ || variances_.size() != k_ || weights_.size() != k_)
      throw std::runtime_error("invalid mixture snapshot");
  }
};
}  // namespace ml
