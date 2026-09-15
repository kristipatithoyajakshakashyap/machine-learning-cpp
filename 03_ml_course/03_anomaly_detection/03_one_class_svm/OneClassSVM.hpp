#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "02_unsupervised/validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// One-class support vector machine (Schoelkopf et al. 2001) with an RBF
// kernel.  The training rows are separated from the origin in feature space
// by a hyperplane w.phi(x) = rho with maximal margin; nu upper-bounds the
// fraction of training rows outside the boundary and lower-bounds the
// fraction of support vectors.  The dual
//     min 1/2 sum_ij a_i a_j K(x_i, x_j)   s.t. 0 <= a_i <= 1/(nu n), sum a = 1
// is solved by a deterministic SMO-style pairwise descent: the most violating
// pair (largest gradient with room to shrink, smallest with room to grow) is
// moved along the equality constraint until the KKT gap closes.
class OneClassSVM {
  static constexpr double kKktTolerance = 1e-6;
  static constexpr size_t kMaxSweeps = 20000;
  double nu_, gamma_requested_, gamma_ = 0, rho_ = 0, threshold_ = 0;
  Mat support_;
  Vec alpha_;
  double kernel(const Vec& a, const Vec& b) const {
    return std::exp(-gamma_ * distance2(a, b));
  }
  double raw_decision(const Vec& q) const {
    double s = 0;
    for (size_t i = 0; i < support_.size(); ++i)
      s += alpha_[i] * kernel(support_[i], q);
    return s;
  }

 public:
  // gamma <= 0 requests the "scale" default 1 / p at fit time.
  explicit OneClassSVM(double nu = .05, double gamma = 0)
      : nu_(nu), gamma_requested_(gamma) {}
  // X must already be preprocessed (imputed, scaled) by the caller.
  void fit(const Mat& X) {
    validate_dense(X);
    const size_t n = X.size();
    if (n < 2) throw std::invalid_argument("at least two training rows required");
    if (!(nu_ > 0) || nu_ > 1) throw std::invalid_argument("nu must lie in (0, 1]");
    gamma_ = gamma_requested_ > 0 ? gamma_requested_ : 1.0 / double(X[0].size());
    Mat K(n, Vec(n, 0));
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i; j < n; ++j) K[i][j] = K[j][i] = kernel(X[i], X[j]);
    const double C = 1.0 / (nu_ * double(n));
    Vec a(n, 1.0 / double(n)), g(n, 0);  // a feasible start; g = K a
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) g[i] += K[i][j] * a[j];
    for (size_t sweep = 0; sweep < kMaxSweeps; ++sweep) {
      size_t up = n, down = n;
      for (size_t i = 0; i < n; ++i) {
        if (a[i] > 0 && (up == n || g[i] > g[up])) up = i;      // can shrink
        if (a[i] < C && (down == n || g[i] < g[down])) down = i;  // can grow
      }
      if (up == n || down == n || g[up] - g[down] < kKktTolerance) break;
      const double curvature =
          std::max(K[up][up] + K[down][down] - 2 * K[up][down], 1e-12);
      const double step =
          std::min({(g[up] - g[down]) / curvature, a[up], C - a[down]});
      a[up] -= step;
      a[down] += step;
      for (size_t i = 0; i < n; ++i) g[i] += step * (K[i][down] - K[i][up]);
    }
    // rho = decision value of free support vectors (0 < a < C); fall back to
    // the mean over all support vectors when none is strictly inside.
    double free_sum = 0, sv_sum = 0;
    size_t free_count = 0, sv_count = 0;
    support_.clear();
    alpha_.clear();
    for (size_t i = 0; i < n; ++i) {
      if (a[i] <= 0) continue;
      support_.push_back(X[i]);
      alpha_.push_back(a[i]);
      sv_sum += g[i];
      ++sv_count;
      if (a[i] < C - 1e-12) {
        free_sum += g[i];
        ++free_count;
      }
    }
    rho_ = free_count ? free_sum / double(free_count) : sv_sum / double(sv_count);
  }
  double nu() const { return nu_; }
  double gamma() const { return gamma_; }
  double rho() const { return rho_; }
  size_t support_count() const { return support_.size(); }
  // Signed distance-like decision: positive inside the learned region.
  Vec decision_function(const Mat& Q) const {
    if (support_.empty()) throw std::logic_error("fit first");
    validate_dense(Q, support_[0].size());
    Vec d;
    d.reserve(Q.size());
    for (const auto& q : Q) d.push_back(raw_decision(q) - rho_);
    return d;
  }
  // Anomaly score = rho - decision: higher means more anomalous, consistent
  // with the Isolation Forest and LOF modules.  Zero is the SVM boundary.
  Vec score_samples(const Mat& Q) const {
    Vec s = decision_function(Q);
    for (double& v : s) v = -v;
    return s;
  }
  void set_threshold(double value) {
    if (!std::isfinite(value))
      throw std::invalid_argument("threshold must be finite");
    threshold_ = value;
  }
  double threshold() const { return threshold_; }
  // 1 = anomaly (score above threshold), 0 = ordinary.  Pass a NaN threshold
  // (the default) to use the stored one; scores may be negative so a
  // sentinel of -1 would be ambiguous.
  Vec predict(const Mat& Q,
              double threshold = std::numeric_limits<double>::quiet_NaN()) const {
    if (std::isnan(threshold)) threshold = threshold_;
    if (!std::isfinite(threshold))
      throw std::invalid_argument("threshold must be finite");
    auto s = score_samples(Q);
    for (double& v : s) v = v > threshold ? 1 : 0;
    return s;
  }
  void save(std::ostream& o) const {
    archive::write(o, std::string("OneClassSVM_V1"), nu_, gamma_, rho_,
                   threshold_, support_, alpha_);
  }
  void load(std::istream& i) {
    std::string tag;
    archive::read(i, tag, nu_, gamma_, rho_, threshold_, support_, alpha_);
    if (tag != "OneClassSVM_V1" || support_.empty() ||
        alpha_.size() != support_.size() || !(gamma_ > 0) || !(nu_ > 0) ||
        nu_ > 1 || !std::isfinite(rho_) || !std::isfinite(threshold_))
      throw std::runtime_error("invalid One-Class SVM snapshot");
    validate_dense(support_);
    gamma_requested_ = gamma_;
  }
};
}  // namespace ml
