#ifndef MLC_MODULE_LINEAR_HPP
#define MLC_MODULE_LINEAR_HPP

// Linear models: ordinary least squares (closed form), ridge and lasso.
// The sklearn equivalents are sklearn.linear_model.LinearRegression, Ridge
// and Lasso.  Weights are stored as w[0]=intercept, w[1..p]=slopes and are
// solved/updated deterministically with the plainest stable arithmetic:
//   - OLS / ridge: normal equations + partial-pivot Gauss elimination;
//   - lasso:       coordinate descent with soft-thresholding on centred
//                  columns (matching sklearn's alpha*n_samples scaling).
//
// The intercept is unpenalised everywhere, and predict() computes
// yhat[i] = w0 + sum_j w_j * X[i][j], so a fitted model can immediately
// score new rows.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

// ---------------------------------------------------------------------------
// LinearRegression: fits OLS / ridge / lasso and predicts.
// ---------------------------------------------------------------------------
class LinearRegression {
public:
  // Closed-form OLS: build augmented design [1, X] and solve
  // (M'M) w = M'y via partial-pivot elimination.  No penalty term.
  void fit_ols(const Mat &X, const Vec &y);

  // Ridge: same normal equations with lambda added to the diagonal of the
  // slope block only (columns 1..p), so the intercept stays unpenalised.
  // Matches sklearn Ridge(alpha=lambda).
  void fit_ridge(const Mat &X, const Vec &y, double lambda);

  // Lasso: coordinate descent + soft threshold on centred columns.
  // Solves (1/(2n))||y - Xw||^2 + lambda*||w[1:]||_1; the n*lambda threshold
  // reproduces sklearn Lasso(alpha=lambda) exactly.  `max_iter`/`tol` bound
  // the sweep schedule / convergence criterion.
  void fit_lasso(const Mat &X, const Vec &y, double lambda,
                 size_t max_iter = 10000, double tol = 1e-6);

  // yhat[i] = w0 + sum_j w_j X[i, j] for a fitted model.
  Vec predict(const Mat &X) const;

  // Model accessors.
  const Vec &coef() const { return w_; }
  double intercept() const { return w_.empty() ? 0.0 : w_[0]; }
  size_t n_features() const { return w_.empty() ? 0 : w_.size() - 1; }

  void set_lambda(double value) {
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("Invalid ridge penalty");
    lambda_ = value;
  }
  void fit(const Mat &X, const Vec &y) { fit_ridge(X, y, lambda_); }
  void save(std::ostream &out) const {
    archive::write(out, std::string("LinearRegression"), w_, lambda_);
  }
  void load(std::istream &in) {
    std::string tag;
    archive::read(in, tag);
    if (tag != "LinearRegression")
      throw std::runtime_error("Wrong model type");
    archive::read(in, w_, lambda_);
    set_lambda(lambda_);
    if (w_.size() < 2)
      throw std::runtime_error("Invalid linear model state");
    for (double v : w_)
      if (!std::isfinite(v))
        throw std::runtime_error("Invalid linear coefficient");
  }

private:
  double lambda_ = 1;
  Vec w_; // [intercept, slopes...]
};

// ---------------------------------------------------------------------------
// PolynomialFeatures: expand p inputs into their d-degree monomials.
// ---------------------------------------------------------------------------
class PolynomialFeatures {
public:
  // degree = highest total degree; include_bias keeps the all-ones column.
  PolynomialFeatures(size_t degree, bool include_bias = true)
      : degree_(degree), include_bias_(include_bias) {}

  // Design matrix for X (n x p -> n x n_output_features(p)).
  Mat transform(const Mat &X) const;

  // Column count sklearn calls n_output_features_.
  size_t n_output_features(size_t p) const;

  void save(std::ostream &out) const {
    archive::write(out, std::string("PolynomialFeatures"), degree_,
                   include_bias_);
  }
  void load(std::istream &in) {
    std::string tag;
    archive::read(in, tag);
    if (tag != "PolynomialFeatures")
      throw std::runtime_error("Wrong model type");
    archive::read(in, degree_, include_bias_);
  }

private:
  size_t degree_;
  bool include_bias_;
};

// Human-readable column names for a polynomial expansion (for printouts).
std::vector<std::string>
polynomial_feature_names(size_t p_in, size_t degree,
                         const std::vector<std::string> &input_names,
                         bool include_bias);

// ---------------------------------------------------------------------------
// Standardisation helpers (z-score).  scale_fit learns mean/std from the
// training set; scale_apply reuses them on new data so the test set never
// influences preprocessing (no data leakage). Population std (ddof=0) matches
// sklearn StandardScaler.
// ---------------------------------------------------------------------------
Mat scale_fit(const Mat &X, Vec *mean, Vec *std);
Mat scale_apply(const Mat &X, const Vec &mean, const Vec &std);
Vec scale_apply(const Vec &, const Vec &mean, const Vec &std); // for single row

// Render fitted weights with feature names (for lesson output).
std::string print_coefs(const LinearRegression &m,
                        const std::vector<std::string> &names, int prec = 7);

} // namespace ml

#endif // ML_LINREG_HPP
