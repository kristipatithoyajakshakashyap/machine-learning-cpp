#pragma once
// 03_ml_course/01_supervised/02_classification/11_lda/Model.hpp
//
// Purpose : Gaussian discriminant analysis for classification. With `quadratic`
//           false this is LDA (one pooled covariance, linear boundaries); with
//           `quadratic` true it is QDA (one covariance per class, quadratic
//           boundaries). 12_qda/Model.hpp derives QDA from this class.
// Inputs  : a feature matrix X (rows x p) and a label vector y. Labels may be any
//           finite doubles; they are sorted and de-duplicated into `labels`.
// Outputs : none written here; callers persist the model through save()/load().
// Run     : compiled into ml_lda (Model.cpp) and ml_qda; used by every lda_* and
//           qda_* target and by tests/model_test.cpp.
//
// Numerical design: instead of inverting each covariance we store its Cholesky
// factor L (cov = L L^T). A triangular solve then gives the Mahalanobis distance
// and the log-determinant is the sum of log(L_ii^2). Ridge `regularization` is
// added to the diagonal so a near-singular covariance still factorises.
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace course {
// Gaussian discriminant analysis; QDA inherits this implementation and selects
// class-specific covariance. Cholesky factors avoid explicit matrix inverses.
//
// Fitted state (all indexed by class position c in `labels`):
//   labels[c]  the original label value          priors[c]  empirical P(class)
//   means[c]   class mean vector (p)              logdet[c]  log det(cov_c)
//   factors[c] lower-triangular Cholesky factor of the (regularised) covariance
// Invariant: after fit(), labels.size() == means.size() == factors.size() >= 2.
class LDA {
public:
  double regularization; // ridge added to every covariance diagonal (> 0)
  bool quadratic;        // false: pooled covariance (LDA); true: per class (QDA)
  ml::Vec labels, priors, logdet;
  ml::Mat means;
  std::vector<ml::Mat> factors;
  // reg: diagonal ridge; q: choose QDA (true) or LDA (false).
  explicit LDA(double reg = 1e-4, bool q = false)
      : regularization(reg), quadratic(q) {}
  // Estimate means, priors and covariance factors from (X, y).
  // Throws std::invalid_argument on empty/ragged/non-finite input, fewer than two
  // classes or a non-positive ridge; std::runtime_error if a covariance is not
  // positive definite even after regularisation.
  // Complexity: O(n p^2) for the scatter matrices plus O(k p^3) for the Cholesky
  // factorisations (k = number of classes).
  void fit(const ml::Mat &X, const ml::Vec &y) {
    if (X.empty() || X[0].empty() || X.size() != y.size() ||
        regularization <= 0)
      throw std::invalid_argument("Invalid discriminant data/regularization");
    const size_t n = X.size(), p = X[0].size();
    for (size_t i = 0; i < n; ++i) {
      if (X[i].size() != p || !std::isfinite(y[i]))
        throw std::invalid_argument("Invalid row");
      for (double v : X[i])
        if (!std::isfinite(v))
          throw std::invalid_argument("Nonfinite feature");
    }
    // Sorted unique labels give each class a stable position c = 0..k-1.
    labels = y;
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (labels.size() < 2)
      throw std::invalid_argument("Need at least two classes");
    const size_t k = labels.size();
    means.assign(k, ml::Vec(p));
    priors.assign(k, 0);
    logdet.assign(k, 0);
    // cls[i] = class position of row i; priors temporarily hold class counts.
    std::vector<size_t> cls(n);
    for (size_t i = 0; i < n; ++i) {
      cls[i] =
          std::lower_bound(labels.begin(), labels.end(), y[i]) - labels.begin();
      priors[cls[i]]++;
      for (size_t j = 0; j < p; ++j)
        means[cls[i]][j] += X[i][j];
    }
    for (size_t c = 0; c < k; ++c)
      for (double &v : means[c])
        v /= priors[c];
    // Per-class scatter matrices sum (x - mean_c)(x - mean_c)^T over class rows.
    std::vector<ml::Mat> cov(k, ml::Mat(p, ml::Vec(p)));
    for (size_t i = 0; i < n; ++i)
      for (size_t a = 0; a < p; ++a)
        for (size_t b = 0; b < p; ++b)
          cov[cls[i]][a][b] +=
              (X[i][a] - means[cls[i]][a]) * (X[i][b] - means[cls[i]][b]);
    if (!quadratic) {
      // LDA: one pooled covariance (total scatter / n) shared by every class.
      ml::Mat pooled(p, ml::Vec(p));
      for (size_t c = 0; c < k; ++c)
        for (size_t a = 0; a < p; ++a)
          for (size_t b = 0; b < p; ++b)
            pooled[a][b] += cov[c][a][b] / n;
      for (auto &m : cov)
        m = pooled;
    } else
      // QDA: maximum-likelihood covariance per class (scatter / class count).
      for (size_t c = 0; c < k; ++c)
        for (auto &row : cov[c])
          for (double &v : row)
            v /= priors[c];
    factors.assign(k, ml::Mat(p, ml::Vec(p)));
    for (size_t c = 0; c < k; ++c) {
      priors[c] /= n; // counts -> empirical class probabilities
      for (size_t a = 0; a < p; ++a)
        cov[c][a][a] += regularization;
      // Cholesky-Banachiewicz: row a of L from rows 0..a-1; logdet accumulates
      // log(L_aa^2) = log(v) because det(cov) = prod L_aa^2.
      for (size_t a = 0; a < p; ++a)
        for (size_t b = 0; b <= a; ++b) {
          double v = cov[c][a][b];
          for (size_t j = 0; j < b; ++j)
            v -= factors[c][a][j] * factors[c][b][j];
          if (a == b) {
            if (v <= 0)
              throw std::runtime_error("Covariance not positive definite");
            factors[c][a][b] = std::sqrt(v);
            logdet[c] += std::log(v);
          } else
            factors[c][a][b] = v / factors[c][b][b];
        }
    }
  }
  // Posterior class probabilities, one row per input row, columns in `labels`
  // order. Throws std::runtime_error if unfitted, std::invalid_argument if a row
  // has the wrong width. Complexity: O(rows * k * p^2).
  ml::Mat predict_proba(const ml::Mat &X) const {
    if (means.empty())
      throw std::runtime_error("Model is not fitted");
    ml::Mat out;
    for (const auto &row : X) {
      if (row.size() != means[0].size())
        throw std::invalid_argument("Feature count mismatch");
      ml::Vec score(labels.size());
      for (size_t c = 0; c < labels.size(); ++c) {
        // Forward substitution L z = (x - mean_c); then z.z is the Mahalanobis
        // distance (x - mean_c)^T cov^-1 (x - mean_c) without forming cov^-1.
        ml::Vec z(row.size());
        for (size_t a = 0; a < row.size(); ++a) {
          double v = row[a] - means[c][a];
          for (size_t b = 0; b < a; ++b)
            v -= factors[c][a][b] * z[b];
          z[a] = v / factors[c][a][a];
        }
        // Log posterior up to a constant: log prior - 0.5 (log det + distance).
        score[c] = std::log(priors[c]) - 0.5 * (logdet[c] + ml::dot(z, z));
      }
      // Log-sum-exp: subtract the peak before exponentiating to avoid overflow.
      double peak = *std::max_element(score.begin(), score.end()), sum = 0;
      for (double &v : score) {
        v = std::exp(v - peak);
        sum += v;
      }
      for (double &v : score)
        v /= sum;
      out.push_back(score);
    }
    return out;
  }
  // Hard labels: the original label value of the most probable class per row.
  ml::Vec predict(const ml::Mat &X) const {
    ml::Vec out;
    for (const auto &row : predict_proba(X))
      out.push_back(
          labels[std::max_element(row.begin(), row.end()) - row.begin()]);
    return out;
  }
  // Serialise a version tag plus every fitted field; the tag distinguishes LDA
  // from QDA archives so a file cannot be loaded into the wrong variant.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string(quadratic ? "QDA_V1" : "LDA_V1"));
    ml::archive::write(out, regularization, quadratic, labels, priors, logdet,
                       means, factors);
  }
  // Inverse of save(); throws std::runtime_error on a tag mismatch.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string(quadratic ? "QDA_V1" : "LDA_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, regularization, quadratic, labels, priors, logdet,
                      means, factors);
  }
};
} // namespace course
