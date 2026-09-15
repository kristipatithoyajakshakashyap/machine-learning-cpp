// 02_data_science/src/pca.cpp
// dsts linear algebra for dimensionality reduction: Jacobi eigen-solver for
// symmetric matrices, a thin SVD built on it, and principal component
// analysis (fit + transform).
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 10_dimensionality, 11_final_pipeline and the math
// foundations lessons. Exercised by ctest -R dsts_regression.
// Matrices are std::vector<std::vector<double>>, row-major.

#include "dsts/pca.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dsts {

// Jacobi rotation sweep for a symmetric matrix; returns eigenvalues (diagonal
// of D) and the accumulated orthogonal matrix V such that A ~ V D V^T.
void symmetric_eigen(const std::vector<std::vector<double>>& A_in,
                     std::vector<double>& eigenvalues,
                     std::vector<std::vector<double>>& V) {
  std::vector<std::vector<double>> A = A_in;
  const size_t n = A.size();
  for (const auto& row : A)
    if (row.size() != n) throw std::invalid_argument("symmetric_eigen() needs a square matrix");
  V.assign(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) V[i][i] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
    }
    // Converged when the off-diagonal energy is negligible; at most 100 sweeps.
    if (off < 1e-24) break;
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        if (std::fabs(A[p][q]) < 1e-24) continue;
        // Classic Jacobi rotation angle chosen to zero A[p][q]; t = tan(angle) picks
        // the smaller root for numerical stability.
        const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        const double t = (theta >= 0.0)
                             ? 1.0 / (theta + std::sqrt(theta * theta + 1.0))
                             : -1.0 / (-theta + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        // rotate A
        for (size_t k = 0; k < n; ++k) {
          const double a_kp = A[k][p], a_kq = A[k][q];
          A[k][p] = c * a_kp - s * a_kq;
          A[k][q] = s * a_kp + c * a_kq;
        }
        for (size_t k = 0; k < n; ++k) {
          const double a_pk = A[p][k], a_qk = A[q][k];
          A[p][k] = c * a_pk - s * a_qk;
          A[q][k] = s * a_pk + c * a_qk;
        }
        // rotate V
        for (size_t k = 0; k < n; ++k) {
          const double v_kp = V[k][p], v_kq = V[k][q];
          V[k][p] = c * v_kp - s * v_kq;
          V[k][q] = s * v_kp + c * v_kq;
        }
      }
    }
  }
  eigenvalues.resize(n);
  for (size_t i = 0; i < n; ++i) eigenvalues[i] = A[i][i];
}

// Thin SVD A (m x n) = U S V^T via the eigen-decomposition of A^T A.
// Singular values are sorted descending. Adequate for small, well-conditioned
// lesson matrices; squaring the condition number makes it unsuitable for
// near-singular inputs.
void svd_thin(const std::vector<std::vector<double>>& A,
              std::vector<std::vector<double>>& U, std::vector<double>& S,
              std::vector<std::vector<double>>& V) {
  if (A.empty() || A[0].empty()) throw std::invalid_argument("svd_thin() needs a non-empty matrix");
  const size_t m = A.size(), n = A[0].size();
  for (const auto& row : A)
    if (row.size() != n) throw std::invalid_argument("svd_thin() ragged matrix");
  // Gram matrix G = A^T A (n x n, symmetric PSD); its eigenvectors are the
  // right singular vectors and its eigenvalues are sigma^2.
  std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (size_t r = 0; r < m; ++r) s += A[r][i] * A[r][j];
      G[i][j] = s;
    }
  std::vector<double> lambda;
  std::vector<std::vector<double>> Q;
  symmetric_eigen(G, lambda, Q);
  std::vector<size_t> order(n);
  for (size_t j = 0; j < n; ++j) order[j] = j;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lambda[a] > lambda[b]; });
  S.assign(n, 0.0);
  V.assign(n, std::vector<double>(n, 0.0));
  U.assign(m, std::vector<double>(n, 0.0));
  for (size_t k = 0; k < n; ++k) {
    const size_t c = order[k];
    S[k] = std::sqrt(std::max(lambda[c], 0.0));
    for (size_t j = 0; j < n; ++j) V[j][k] = Q[j][c];
    if (S[k] <= 1e-300) continue;  // rank-deficient column: leave U column zero
    // u_k = A v_k / sigma_k
    for (size_t r = 0; r < m; ++r) {
      double s = 0.0;
      for (size_t j = 0; j < n; ++j) s += A[r][j] * V[j][k];
      U[r][k] = s / S[k];
    }
  }
}

// Fit PCA on X (n rows, p features). Rows with any non-finite value are
// dropped. Returns column means, eigenvalues (descending, clamped at 0),
// explained-variance ratios, loadings (row i = PC i) and scores for the kept
// rows. Requires n >= 2 and p >= 2.
PcaResult pca(const std::vector<std::vector<double>>& X) {
  if (X.empty()) throw std::invalid_argument("pca() needs rows");
  const size_t n = X.size(), p = X[0].size();
  if (n < 2 || p < 2) throw std::invalid_argument("pca() needs n>=2 and p>=2");

  // Listwise deletion of rows with any NaN.
  std::vector<size_t> keep;
  for (size_t i = 0; i < n; ++i) {
    if (X[i].size() != p) throw std::invalid_argument("pca() ragged matrix");
    bool ok = true;
    for (double v : X[i]) ok = ok && std::isfinite(v);
    if (ok) keep.push_back(i);
  }
  const size_t m = keep.size();
  if (m < 2) throw std::invalid_argument("pca() has < 2 complete rows");

  // Centre columns.
  std::vector<double> mu(p, 0.0);
  for (size_t i : keep) {
    for (size_t j = 0; j < p; ++j) mu[j] += X[i][j];
  }
  for (double& v : mu) v /= static_cast<double>(m);

  std::vector<std::vector<double>> Xc(m, std::vector<double>(p));
  for (size_t r = 0; r < m; ++r) {
    for (size_t j = 0; j < p; ++j) {
      Xc[r][j] = X[keep[r]][j] - mu[j];
    }
  }

  // Covariance matrix C = Xc^T Xc / (m - 1).
  std::vector<std::vector<double>> C(p, std::vector<double>(p, 0.0));
  for (size_t a = 0; a < p; ++a) {
    for (size_t b = 0; b < p; ++b) {
      double s = 0.0;
      for (size_t r = 0; r < m; ++r) s += Xc[r][a] * Xc[r][b];
      C[a][b] = s / static_cast<double>(m - 1);
    }
  }

  PcaResult res;
  res.mean = mu;
  std::vector<std::vector<double>> V;
  symmetric_eigen(C, res.eigenvalues, V);

  // Sort eigenvectors by descending eigenvalue; V column j holds PC j.
  std::vector<size_t> order(p);
  for (size_t j = 0; j < p; ++j) order[j] = j;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return res.eigenvalues[a] > res.eigenvalues[b];
  });
  std::vector<double> eig(p), ratio(p);
  std::vector<std::vector<double>> loadings(p, std::vector<double>(p));
  double total = 0.0;
  for (size_t a : order) total += res.eigenvalues[a];
  for (size_t i = 0; i < p; ++i) {
    const size_t a = order[i];
    eig[i] = std::max(res.eigenvalues[a], 0.0);
    ratio[i] = total > 0.0 ? eig[i] / total : 0.0;
    for (size_t j = 0; j < p; ++j) loadings[i][j] = V[j][a];
  }
  res.eigenvalues = eig;
  res.explained_ratio = ratio;
  res.loadings = loadings;

  // Scores: Xc * V_k (all PCs).
  res.scores.assign(m, std::vector<double>(p, 0.0));
  for (size_t r = 0; r < m; ++r) {
    for (size_t i = 0; i < p; ++i) {
      double s = 0.0;
      for (size_t j = 0; j < p; ++j) s += Xc[r][j] * loadings[i][j];
      res.scores[r][i] = s;
    }
  }
  return res;
}

// Project new rows onto a fitted PCA: (x - mean) . loadings[k] per component.
// Validates the fitted shape and that every input is finite.
std::vector<std::vector<double>> pca_transform(const PcaResult& fitted,
    const std::vector<std::vector<double>>& X) {
  const size_t p = fitted.mean.size();
  if (p == 0 || fitted.loadings.size() != p) throw std::invalid_argument("PCA is not fitted");
  for (const auto& axis : fitted.loadings)
    if (axis.size() != p) throw std::invalid_argument("invalid PCA axis");
  std::vector<std::vector<double>> scores(X.size(), std::vector<double>(p));
  for (size_t r = 0; r < X.size(); ++r) {
    if (X[r].size() != p) throw std::invalid_argument("PCA feature count mismatch");
    for (size_t j = 0; j < p; ++j) {
      if (!std::isfinite(X[r][j])) throw std::invalid_argument("PCA transform needs finite values");
      for (size_t k = 0; k < p; ++k)
        scores[r][k] += (X[r][j] - fitted.mean[j]) * fitted.loadings[k][j];
    }
  }
  return scores;
}
}  // namespace dsts
