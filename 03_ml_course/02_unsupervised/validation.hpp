// 03_ml_course/02_unsupervised/validation.hpp
// Purpose: tiny shared guards used by every unsupervised and anomaly-detection
//          model (k-means, agglomerative, DBSCAN, GMM, PCA, t-SNE, IF, LOF, OC-SVM).
// Inputs:  none (header only; included by the module headers and .cpp files).
// Outputs: none (prints nothing, writes nothing).
// Run target: not an executable; compiled into every module that includes it.
#pragma once
#include <cmath>
#include <stdexcept>

#include "helper/math/matrix.hpp"
namespace ml {
// Reject matrices the fitting code cannot handle.
// Parameters: X - dense row-major feature matrix; p - expected feature count
//             (0 means "use the width of the first row").
// Throws std::invalid_argument when X is empty, ragged, or holds NaN/inf.
// Invariant: after this returns, every row has exactly p finite values, so
// callers can index X[i][j] without further checks. Complexity O(n*p).
inline void validate_dense(const Mat& X, size_t p = 0) {
  if (X.empty() || X[0].empty())
    throw std::invalid_argument("nonempty matrix required");
  if (!p) p = X[0].size();
  for (const auto& row : X) {
    if (row.size() != p)
      throw std::invalid_argument("feature dimension mismatch");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite features required; impute first");
  }
}
// Squared Euclidean distance ||a - b||^2 (no sqrt: cheaper and order-preserving,
// which is all nearest-neighbour searches need). Assumes a.size() == b.size().
// Complexity O(p).
inline double distance2(const Vec& a, const Vec& b) {
  double s = 0;
  for (size_t j = 0; j < a.size(); ++j) s += (a[j] - b[j]) * (a[j] - b[j]);
  return s;
}
}  // namespace ml
