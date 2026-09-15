// 03_ml_course/02_unsupervised/06_pca/PCA.hpp
// Purpose: self-contained principal component analysis used by the upca_*
//          lessons, upca_predict and tests/verify_reload.cpp. Header-only.
// Inputs:  a dense finite feature matrix passed to fit(); the number of
//          components to keep is the only hyper-parameter.
// Outputs: none directly (no printing, no files); callers use transform(),
//          inverse_transform(), explained_variance_ratio(), components() and
//          persist the object with save()/load().
// Run target: not an executable; linked via the ml_pca INTERFACE library.
//
// Method: centre the data, form the p x p sample covariance matrix, and
// diagonalise it with cyclic Jacobi rotations (exact for symmetric matrices,
// trivially small for p = 4 features). Eigenvectors sorted by decreasing
// eigenvalue are the principal axes; projecting centred rows onto the first
// `components` axes gives the scores.
// sklearn equivalent: sklearn.decomposition.PCA(n_components=components,
//   svd_solver='full'). Signs of individual axes may differ (both are valid).
#pragma once
#include <algorithm>
#include <numeric>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// Symmetric covariance eigendecomposition by Jacobi plane rotations.
class PCA {
  size_t components_;       // number of axes kept (1..p)
  Vec mean_, eigenvalues_;  // feature means (p); all p eigenvalues, descending
  Mat axes_;                // components_ x p unit-length principal axes

 public:
  explicit PCA(size_t components = 2) : components_(components) {}
  // Fit the mean, covariance eigen-pairs and the leading axes. Deterministic.
  // Complexity: O(n p^2) for the covariance plus O(p^3) per Jacobi sweep.
  // Throws on fewer than 2 rows or an invalid component count.
  void fit(const Mat& X) {
    validate_dense(X);
    size_t p = X[0].size();
    if (X.size() < 2 || !components_ || components_ > p)
      throw std::invalid_argument("invalid PCA dimensions");
    // Column means (used for centring in transform / inverse_transform).
    mean_ = Vec(p, 0);
    for (const auto& r : X)
      for (size_t j = 0; j < p; ++j) mean_[j] += r[j] / X.size();
    // Sample covariance C (divides by n - 1) and V = identity, which will
    // accumulate the rotations and end up holding the eigenvectors as columns.
    Mat C(p, Vec(p, 0)), V(p, Vec(p, 0));
    for (size_t a = 0; a < p; ++a) {
      V[a][a] = 1;
      for (size_t b = 0; b < p; ++b)
        for (const auto& r : X)
          C[a][b] +=
              (r[a] - mean_[a]) * (r[b] - mean_[b]) / double(X.size() - 1);
    }
    // Classical Jacobi: repeatedly zero the largest off-diagonal element with
    // a plane rotation in the (a, b) plane. Each rotation preserves the
    // eigenvalues and pushes the matrix closer to diagonal. The iteration cap
    // 100 p^2 is far above what convergence needs for small p.
    for (size_t t = 0; t < 100 * p * p; ++t) {
      double largest = 0;
      size_t a = 0, b = 0;
      for (size_t i = 0; i < p; ++i)
        for (size_t j = i + 1; j < p; ++j)
          if (std::abs(C[i][j]) > largest) {
            largest = std::abs(C[i][j]);
            a = i;
            b = j;
          }
      if (largest < 1e-12) break;  // effectively diagonal: converged
      // Rotation angle that annihilates C[a][b].
      double angle = .5 * std::atan2(2 * C[a][b], C[b][b] - C[a][a]),
             c = std::cos(angle), s = std::sin(angle), aa = C[a][a],
             bb = C[b][b], ab = C[a][b];
      // Update the rows/columns a and b that are not on the 2x2 block.
      for (size_t j = 0; j < p; ++j)
        if (j != a && j != b) {
          double x = C[j][a], y = C[j][b];
          C[j][a] = C[a][j] = c * x - s * y;
          C[j][b] = C[b][j] = s * x + c * y;
        }
      // Closed-form update of the 2x2 block; the off-diagonal becomes 0.
      C[a][a] = c * c * aa - 2 * s * c * ab + s * s * bb;
      C[b][b] = s * s * aa + 2 * s * c * ab + c * c * bb;
      C[a][b] = C[b][a] = 0;
      // Apply the same rotation to the eigenvector accumulator.
      for (size_t j = 0; j < p; ++j) {
        double x = V[j][a], y = V[j][b];
        V[j][a] = c * x - s * y;
        V[j][b] = s * x + c * y;
      }
    }
    // Order axes by decreasing eigenvalue (= diagonal of the rotated C).
    std::vector<size_t> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return C[a][a] > C[b][b]; });
    eigenvalues_.clear();
    axes_.clear();
    // Clamp tiny negative round-off eigenvalues to 0.
    for (size_t j : order) eigenvalues_.push_back(std::max(0.0, C[j][j]));
    // Keep only the leading `components_` eigenvectors (columns of V).
    for (size_t c = 0; c < components_; ++c) {
      Vec axis(p);
      for (size_t j = 0; j < p; ++j) axis[j] = V[j][order[c]];
      axes_.push_back(axis);
    }
  }
  // Project rows onto the kept axes: z = (x - mean) . axis_c. Returns an
  // n x components matrix of scores. O(n components p).
  Mat transform(const Mat& X) const {
    if (axes_.empty()) throw std::logic_error("fit first");
    validate_dense(X, mean_.size());
    Mat z(X.size(), Vec(components_, 0));
    for (size_t i = 0; i < X.size(); ++i)
      for (size_t c = 0; c < components_; ++c)
        for (size_t j = 0; j < mean_.size(); ++j)
          z[i][c] += (X[i][j] - mean_[j]) * axes_[c][j];
    return z;
  }
  // Map scores back to feature space: x_hat = mean + sum_c z_c * axis_c.
  // Exact only when components == p; otherwise the discarded variance is lost.
  Mat inverse_transform(const Mat& z) const {
    if (axes_.empty()) throw std::logic_error("fit first");
    validate_dense(z, components_);
    Mat X(z.size(), mean_);
    for (size_t i = 0; i < z.size(); ++i)
      for (size_t c = 0; c < components_; ++c)
        for (size_t j = 0; j < mean_.size(); ++j)
          X[i][j] += z[i][c] * axes_[c][j];
    return X;
  }
  // Fraction of total variance carried by each of the p axes (sums to 1),
  // including axes that were not kept, so a full scree plot is possible.
  Vec explained_variance_ratio() const {
    double sum = std::accumulate(eigenvalues_.begin(), eigenvalues_.end(), 0.0);
    Vec v = eigenvalues_;
    for (double& x : v) x = sum > 0 ? x / sum : 0;
    return v;
  }
  // The kept principal axes (loadings), one unit vector per row.
  const Mat& components() const { return axes_; }
  // Persist the component count, mean, eigenvalues and axes.
  void save(std::ostream& o) const {
    archive::write(o, components_, mean_, eigenvalues_, axes_);
  }
  // Inverse of save(); checks the axes are p-wide and exactly components_ many.
  void load(std::istream& i) {
    archive::read(i, components_, mean_, eigenvalues_, axes_);
    validate_dense(axes_, mean_.size());
    if (axes_.size() != components_)
      throw std::runtime_error("invalid PCA snapshot");
  }
};
}  // namespace ml
