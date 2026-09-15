#ifndef MLC_MODULE_NAIVEBAYES_HPP
#define MLC_MODULE_NAIVEBAYES_HPP

// Gaussian Naive Bayes classifier.
// The sklearn equivalent is sklearn.naive_bayes.GaussianNB.
//
// Assumes each feature is independently Gaussian-distributed within each
// class.  Fit computes per-class mean (mu_k) and variance (sigma_k^2) for
// every feature.  Predict evaluates the log-posterior for each class k:
//
//   log p(k|x) = log pi_k  +  sum_j  log N(x_j | mu_kj, sigma_kj^2)
//
// where pi_k is the class prior (fraction of training rows with label k)
// and N is the univariate Gaussian density.  Variance is regularised by
// var_smoothing (sklearn default 1e-9) to avoid log(0).
//
// Uses population variances. A machine-epsilon variance floor keeps entirely
// constant datasets well-defined when relative smoothing alone would vanish.

#include <cstddef>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

class GaussianNB {
 public:
  // var_smoothing: fraction of the largest variance of any feature added to
  // stability. Must be finite and nonnegative. A machine-epsilon floor also
  // applies, including when smoothing is zero.
  explicit GaussianNB(double var_smoothing = 1e-9);

  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;

  // Model accessors for lesson inspection.
  const Mat& class_mean() const { return class_mean_; }
  const Mat& class_var() const { return class_var_; }
  const Vec& class_prior() const { return class_prior_; }
  size_t n_classes() const { return n_classes_; }

  void save(std::ostream& out) const;
  void load(std::istream& in);
 private:
  void validate_state() const;
  void validate_rows(const Mat& X) const;
  double var_smoothing_;
  size_t n_classes_ = 0;
  Vec class_prior_;          // pi_k
  Mat class_mean_;           // [k][j] = mean of feature j in class k
  Mat class_var_;            // [k][j] = variance of feature j in class k
  std::vector<double> classes_;
};

}  // namespace ml

#endif  // ML_NBAYES_HPP
