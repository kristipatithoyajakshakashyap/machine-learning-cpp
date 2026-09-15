#ifndef DSTS_PCA_HPP
#define DSTS_PCA_HPP

// Mean-centred PCA via the eigen-decomposition of the sample covariance
// matrix (Jacobi rotations). sklearn equivalent: sklearn.decomposition.PCA
// with svd_solver="full" on mean-centred data.

#include <string>
#include <vector>

namespace dsts {

struct PcaResult {
  std::vector<double> mean;                  // training centering, reused at inference
  std::vector<double> eigenvalues;            // sorted descending
  std::vector<std::vector<double>> loadings;  // p x p: loading[i][j] = PC i,
                                              // variable j (each row is a PC)
  std::vector<double> explained_ratio;        // eigenvalue / sum(eigenvalues)
  std::vector<std::vector<double>> scores;    // n x p projected data (all PCs)
};

// Eigen-decomposition of a symmetric n x n matrix by cyclic Jacobi rotations.
// On return values[i] is the i-th eigenvalue (unsorted) and column i of
// vectors (vectors[row][i]) is its unit eigenvector, so A ~ V diag(values) V^T.
// numpy equivalent: numpy.linalg.eigh (up to ordering).
void symmetric_eigen(const std::vector<std::vector<double>>& A,
                     std::vector<double>& values,
                     std::vector<std::vector<double>>& vectors);

// Thin SVD of an m x n matrix (m >= n expected) via the eigen-decomposition of
// A^T A: A = U diag(S) V^T with U (m x n), S (n, descending), V (n x n).
// Columns of U/V are indexed as U[row][k]. numpy: numpy.linalg.svd(A, full_matrices=False).
void svd_thin(const std::vector<std::vector<double>>& A,
              std::vector<std::vector<double>>& U, std::vector<double>& S,
              std::vector<std::vector<double>>& V);

// X is an n x p dense matrix (rows = observations). Non-finite rows are
// skipped; ragged matrices are rejected. Requires >=2 finite rows and p>=2.
PcaResult pca(const std::vector<std::vector<double>>& X);
std::vector<std::vector<double>> pca_transform(const PcaResult& fitted,
    const std::vector<std::vector<double>>& X);

}  // namespace dsts

#endif  // DSTS_PCA_HPP
