#include "LinearRegression.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ml {

// ---------------------------------------------------------------------------
// Linear regression toolkit
//
// LinearRegression fits ordinary least squares (closed form), ridge and
// lasso.  All three minimise a sum of squared errors and differ only in the
// penalty placed on the slopes:
//   OLS    min ||y - X w||^2                                  (no penalty)
//   ridge  min ||y - X w||^2 + lambda * ||w[1:]||^2           (L2)
//   lasso  min (1/2n)||y - X w||^2 + lambda * ||w[1:]||_1     (L1)
// The intercept (w[0]) is never penalised.  Weights are stored as
// w[0]=intercept, w[1..p]=slopes.  Everything is solved with plain double
// arithmetic and deterministic code paths so the exact same inputs always
// give the exact same output (matching sklearn's numbers to 7 digits).
// ---------------------------------------------------------------------------

// ---- internal helpers ----------------------------------------------------
namespace {
void validate_rows(const Mat &X, size_t p) {
  for (const auto &row : X) {
    if (row.size() != p)
      throw std::invalid_argument("linear: feature count mismatch");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("linear: nonfinite feature");
  }
}
void validate_training(const Mat &X, const Vec &y) {
  if (X.empty() || X[0].empty() || X.size() != y.size())
    throw std::invalid_argument("linear: training dimensions invalid");
  validate_rows(X, X[0].size());
  for (double value : y)
    if (!std::isfinite(value))
      throw std::invalid_argument("linear: nonfinite target");
}

// Augmented design: M[i][0] = 1 (intercept column) and M[i][j+1] = X[i][j].
// Building a code, because the linear algebra helpers contract on Mat/Vec,
// and the intercept is just another column for the normal equations.
struct Aug {
  Mat M;
  size_t n, p; // n observations, p features including intercept col
};

Aug make_aug(const Mat &X) {
  Aug a;
  a.n = X.size();
  a.p = X.empty() ? 0 : X[0].size() + 1; // +1 for bias col
  a.M.resize(a.n, Vec(a.p, 1.0));        // col 0 = 1
  for (size_t i = 0; i < a.n; ++i)
    for (size_t j = 0; j < a.p - 1; ++j)
      a.M[i][j + 1] = X[i][j];
  return a;
}

// Solve the OLS normal equations (M' M) w = M' y by partial-pivot Gauss
// elimination.  The (X'X) product is assembled in the upper triangle then
// mirrored to the lower one (the `k != j` guard avoids double-counting the
// diagonal), which keeps the matrix exactly symmetric.
Vec ols_solve(const Aug &a, const Vec &y) {
  size_t p = a.p;
  Mat AtA(p, Vec(p, 0.0));
  Vec Aty(p, 0.0);
  for (size_t i = 0; i < a.n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      Aty[j] += a.M[i][j] * y[i];
      for (size_t k = j; k < p; ++k) {
        double v = a.M[i][j] * a.M[i][k];
        AtA[j][k] += v;
        if (k != j)
          AtA[k][j] += v;
      }
    }
  }
  return solve_linear(AtA, Aty);
}

// Soft-thresholding operator used by lasso's coordinate descent:
//   S(z; g) = sign(z) * max(|z| - g, 0).
// It is the closed-form solution of the one-dimensional L1-regularised
// problem and is what drives coefficients exactly to zero for small |z|.
double soft(double z, double g) {
  if (z > g)
    return z - g;
  if (z < -g)
    return z + g;
  return 0.0;
}

// Pretty name for a monomial like "x0 x1^2 x3" (powers skipped when 1).
// Feeds PolynomialFeatures' column labels so lesson output is readable.
std::string monomial_name(size_t p, const std::vector<size_t> &exps,
                          const std::vector<std::string> &feat) {
  std::ostringstream o;
  for (size_t j = 0; j < p; ++j) {
    if (exps[j] == 0)
      continue;
    if (!o.str().empty())
      o << " ";
    std::string nm = (j < feat.size()) ? feat[j] : ("x" + std::to_string(j));
    o << nm;
    if (exps[j] > 1)
      o << "^" << exps[j];
  }
  return o.str();
}

// Enumerate every exponent tuple with sum(exponents) <= deg, i.e. all
// monomials of total degree up to `deg` over p inputs.  Ordering matches
// sklearn's PolynomialFeatures: by total degree, then reverse-lex so
// x0^2 comes before x0 x1 before x1^2.  Deterministic -> stable columns.
std::vector<std::vector<size_t>> enumerate_monomials(size_t p, size_t deg) {
  std::vector<std::vector<size_t>> out;
  std::vector<size_t> e(p, 0);
  std::function<void(size_t, size_t)> dfs = [&](size_t idx, size_t left) {
    if (idx == p) {
      out.push_back(e);
      return;
    }
    for (size_t d = 0; d <= left; ++d) {
      e[idx] = d;
      dfs(idx + 1, left - d);
    }
  };
  dfs(0, deg);
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    size_t sa = 0, sb = 0;
    for (size_t v : a)
      sa += v;
    for (size_t v : b)
      sb += v;
    if (sa != sb)
      return sa < sb;
    // within same degree, reverse lex so x0-x1-x2 order
    return a > b;
  });
  return out;
}

} // namespace

// ---- LinearRegression ---------------------------------------------------

// OLS: closed form via normal equations on the augmented design.
void LinearRegression::fit_ols(const Mat &X, const Vec &y) {
  validate_training(X, y);
  if (X.empty() || X.size() != y.size())
    throw std::runtime_error("fit_ols: dimension mismatch");
  w_ = ols_solve(make_aug(X), y);
}

// Ridge: same normal equations but add lambda to every diagonal entry of the
// slope block (columns j = 1..p).  The intercept column stays untouched so
// the intercept is not shrunk toward zero.  (X'X + lambda*I) w = X'y has a
// unique solution for any lambda > 0, which is what makes ridge stable even
// when X'X is singular (multicollinearity).
void LinearRegression::fit_ridge(const Mat &X, const Vec &y, double lambda) {
  validate_training(X, y);
  if (!std::isfinite(lambda) || lambda < 0)
    throw std::invalid_argument("Invalid ridge penalty");
  if (X.empty() || X.size() != y.size())
    throw std::runtime_error("fit_ridge: dimension mismatch");
  auto a = make_aug(X);
  size_t p = a.p;
  Mat AtA(p, Vec(p, 0.0));
  Vec Aty(p, 0.0);
  for (size_t i = 0; i < a.n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      Aty[j] += a.M[i][j] * y[i];
      for (size_t k = j; k < p; ++k) {
        double v = a.M[i][j] * a.M[i][k];
        AtA[j][k] += v;
        if (k != j)
          AtA[k][j] += v;
      }
    }
  }
  for (size_t j = 1; j < p; ++j)
    AtA[j][j] += lambda;
  w_ = solve_linear(AtA, Aty);
}

// Lasso by coordinate descent.
//
// sklearn fits Lasso on *centred* columns and then recovers the intercept at
// the end (intercept_ = y_mean - X_mean . coef_).  We mirror that exactly:
//   1. centre X and y (subtract per-column / y mean);
//   2. sweep coordinates: each w_j is minimised while all others are fixed,
//      w_j = soft( Xc_j . r_partial , n*lambda ) / ||Xc_j||^2;
//   3. stop when the largest single-coordinate change drops below tol.
// The n*lambda threshold reproduces sklearn's internal alpha*n_samples
// scaling of its L1 coefficient, so Lasso(alpha=lambda) numbers match to
// ~1e-10 when both converge tightly.
void LinearRegression::fit_lasso(const Mat &X, const Vec &y, double lambda,
                                 size_t max_iter, double tol) {
  validate_training(X, y);
  if (!std::isfinite(lambda) || lambda < 0 || max_iter == 0 ||
      !std::isfinite(tol) || tol <= 0)
    throw std::invalid_argument("Invalid lasso optimization parameters");
  if (X.empty() || X.size() != y.size())
    throw std::runtime_error("fit_lasso: dimension mismatch");
  size_t n = X.size();
  size_t p = X[0].size();

  // Centre the design and the target (the intercept is implicit in the means).
  Vec mu_x(p, 0.0);
  double mu_y = 0.0;
  for (size_t i = 0; i < n; ++i)
    mu_y += y[i];
  mu_y /= static_cast<double>(n);
  Vec yc(n);
  for (size_t i = 0; i < n; ++i)
    yc[i] = y[i] - mu_y;
  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < n; ++i)
      mu_x[j] += X[i][j];
    mu_x[j] /= static_cast<double>(n);
  }
  Mat Xc(n, Vec(p));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < p; ++j)
      Xc[i][j] = X[i][j] - mu_x[j];

  // Coordinate descent on the centred problem:
  //   min_w  (1/(2n))|| yc - Xc w ||^2 + lambda*||w||_1
  // The gradient-matching stationarity condition per coordinate is
  //   w_j = soft_threshold( Xc_j . r_partial , n*lambda ) / ||Xc_j||^2,
  // which is exactly sklearn's alpha*n_samples scaling (their l1_reg).
  w_.assign(p + 1, 0.0); // w_[0] holds intercept; slopes updated below
  Vec w(p, 0.0);         // slopes (temporary vector, separate from w_[0])
  Vec r = yc;            // residual of the centred problem
  Vec z(p, 0.0);         // z[j] = ||Xc_j||^2, constant across sweeps
  for (size_t j = 0; j < p; ++j)
    for (size_t i = 0; i < n; ++i)
      z[j] += Xc[i][j] * Xc[i][j];

  for (size_t iter = 0; iter < max_iter; ++iter) {
    double max_c = 0.0; // largest |coordinate change| this sweep
    for (size_t j = 0; j < p; ++j) {
      if (z[j] < 1e-300)
        continue; // constant column: nothing to fit
      double rho = 0.0;
      for (size_t i = 0; i < n; ++i)
        rho += Xc[i][j] * r[i];
      rho += w[j] * z[j]; // add back w_j*z_j -> rho = Xc_j . r_partial
      double w_new = soft(rho, static_cast<double>(n) * lambda) / z[j];
      double delta = w_new - w[j];
      if (std::fabs(delta) > max_c)
        max_c = std::fabs(delta);
      w[j] = w_new;
      // Update the residual in place: only this column's contribution moved.
      for (size_t i = 0; i < n; ++i)
        r[i] -= delta * Xc[i][j];
    }
    if (max_c < tol)
      break; // converged (each coordinate changed < tol)
  }

  // Recover intercept: y_mean - w . x_mean (exactly sklearn's
  // intercept_ = y_offset - X_offset . coef_).
  w_[0] = mu_y;
  for (size_t j = 0; j < p; ++j)
    w_[j + 1] = w[j];
  for (size_t j = 0; j < p; ++j)
    w_[0] -= w[j] * mu_x[j];
}

// Prediction: yhat[i] = w0 + sum_j w_j * X[i][j]  (dot product per row).
Vec LinearRegression::predict(const Mat &X) const {
  if (w_.empty())
    throw std::runtime_error("predict: model not fitted");
  validate_rows(X, w_.size() - 1);
  size_t n = X.size();
  size_t p = X.empty() ? 0 : X[0].size();
  Vec yhat(n, w_[0]);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < p; ++j)
      yhat[i] += w_[j + 1] * X[i][j];
  return yhat;
}

// ---- PolynomialFeatures ------------------------------------------------

// Expand X into its polynomial design matrix: every monomial of total degree
// <= degree, in enumerate_monomials order.  Removing the first column
// (the all-ones bias) when include_bias is false lets the caller fit with an
// explicit intercept via LinearRegression.
Mat PolynomialFeatures::transform(const Mat &X) const {
  if (X.empty())
    return {};
  validate_rows(X, X[0].size());
  size_t n = X.size();
  size_t p = X[0].size();
  auto exps = enumerate_monomials(p, degree_);
  Mat out(n, Vec(exps.size()));
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < exps.size(); ++c) {
      double v = 1.0;
      for (size_t j = 0; j < p; ++j) {
        for (size_t k = 0; k < exps[c][j]; ++k)
          v *= X[r][j];
      }
      out[r][c] = v;
    }
  }
  if (!include_bias_ && !out.empty() && !out[0].empty()) {
    for (auto &row : out)
      row.erase(row.begin());
  }
  return out;
}

// Number of output columns for p inputs (sklearn's n_output_features_).
size_t PolynomialFeatures::n_output_features(size_t p) const {
  return enumerate_monomials(p, degree_).size() - (include_bias_ ? 0 : 1);
}

// Column names for the polynomial expansion (used in lesson printouts).
std::vector<std::string>
polynomial_feature_names(size_t p_in, size_t degree,
                         const std::vector<std::string> &input_names,
                         bool include_bias) {
  auto exps = enumerate_monomials(p_in, degree);
  std::vector<std::string> out;
  if (!include_bias && !exps.empty())
    exps.erase(exps.begin());
  for (const auto &e : exps)
    out.push_back(monomial_name(p_in, e, input_names));
  return out;
}

// ---- scale --------------------------------------------------------------
// z-score standardisation helpers.  Mean/std are computed from the training
// set only (scale_fit), then scale_apply reuses them on the test set so the
// test data is not used to inform preprocessing (no data leakage).

// Compute per-column mean/std and return the standardised matrix.
Mat scale_fit(const Mat &X, Vec *mean, Vec *std) {
  if (X.empty() || X[0].empty())
    throw std::invalid_argument("scale_fit needs rows and features");
  validate_rows(X, X[0].size());
  size_t n = X.size(), p = X[0].size();
  Vec m(p, 0.0), s(p, 0.0);
  for (size_t j = 0; j < p; ++j) { // column means
    for (size_t i = 0; i < n; ++i)
      m[j] += X[i][j];
    m[j] /= static_cast<double>(n);
  }
  for (size_t j = 0; j < p; ++j) { // population std (ddof=0)
    for (size_t i = 0; i < n; ++i)
      s[j] += (X[i][j] - m[j]) * (X[i][j] - m[j]);
    s[j] = std::sqrt(s[j] / static_cast<double>(n));
  }
  if (mean)
    *mean = m;
  if (std)
    *std = s;
  return scale_apply(X, m, s);
}

// Apply a previously-fit (mean, std) to new data.
Mat scale_apply(const Mat &X, const Vec &mean, const Vec &std) {
  if (mean.empty() || mean.size() != std.size())
    throw std::invalid_argument("Invalid scaling dimensions");
  validate_rows(X, mean.size());
  Mat Z = X;
  size_t n = X.size(), p = mean.size();
  for (size_t j = 0; j < p; ++j) {
    if (!std::isfinite(mean[j]) || !std::isfinite(std[j]) || std[j] < 0)
      throw std::invalid_argument("Invalid scaling parameter");
    for (size_t i = 0; i < n; ++i)
      Z[i][j] = (X[i][j] - mean[j]) / (std[j] > 1e-15 ? std[j] : 1);
  }
  return Z;
}

// Single-row (or single-column) version of scale_apply.
Vec scale_apply(const Vec &v, const Vec &mean, const Vec &std) {
  return scale_apply(Mat{v}, mean, std).front();
}

// Render fitted weights with human-readable feature names for lesson output.
std::string print_coefs(const LinearRegression &m,
                        const std::vector<std::string> &names, int prec) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(prec);
  o << "intercept " << m.intercept();
  const auto &w = m.coef();
  for (size_t j = 1; j < w.size(); ++j) {
    o << "  " << (j - 1 < names.size() ? names[j - 1] : "x" + std::to_string(j))
      << " " << w[j];
  }
  return o.str();
}

} // namespace ml
