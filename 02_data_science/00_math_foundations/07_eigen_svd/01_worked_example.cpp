// 00_math_foundations/07_eigen_svd/01_worked_example.cpp
// Eigenpairs of a symmetric 3x3 matrix and a thin SVD of a 4x2 matrix.
//
// WHAT YOU LEARN:
//   - dsts::symmetric_eigen returns eigenvalues and unit eigenvectors; each
//     pair is verified with the residual ||A v - lambda v||.
//   - The eigenvalues sum to trace(A) and the eigenvectors are orthonormal.
//   - dsts::svd_thin gives A = U diag(S) V^T; the reconstruction error and
//     the rank-1 truncation error are both computed explicitly.
//
// EXPECTED OUTPUT:
//   A (3x3 symmetric) eigenvalues (desc): <three values summing to 9>
//   residual ||A v - lambda v|| per pair < 1e-9
//   thin SVD of 4x2: singular values, reconstruction error < 1e-9
//   wrote worked_example.csv ; exit 0 when all checks hold

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dsts/pca.hpp"

using Matrix = std::vector<std::vector<double>>;

std::vector<double> mat_vec(const Matrix& A, const std::vector<double>& v) {
  std::vector<double> out(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < v.size(); ++j) out[i] += A[i][j] * v[j];
  return out;
}

double norm(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

// Rebuild A from the first k singular triplets.
Matrix reconstruct(const Matrix& U, const std::vector<double>& S, const Matrix& V, size_t k) {
  Matrix R(U.size(), std::vector<double>(V.size(), 0.0));
  for (size_t i = 0; i < U.size(); ++i)
    for (size_t j = 0; j < V.size(); ++j)
      for (size_t c = 0; c < k; ++c) R[i][j] += U[i][c] * S[c] * V[j][c];
  return R;
}

double frobenius_diff(const Matrix& A, const Matrix& B) {
  double s = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < A[i].size(); ++j) s += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]);
  return std::sqrt(s);
}

int main() {
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(RUN_OUTPUT_DIR "/worked_example.csv");
  out << std::setprecision(17);
  std::cout << std::setprecision(10);
  int failures = 0;

  // ---- symmetric eigen ----------------------------------------------------
  const Matrix A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
  std::vector<double> lambda;
  Matrix V;
  dsts::symmetric_eigen(A, lambda, V);
  std::vector<size_t> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lambda[a] > lambda[b]; });

  std::cout << "A = [[4,1,0],[1,3,1],[0,1,2]]  trace = 9\n";
  out << "kind,index,value,residual\n";
  double trace_sum = 0.0;
  for (size_t k = 0; k < 3; ++k) {
    const size_t c = order[k];
    std::vector<double> v(3);
    for (size_t i = 0; i < 3; ++i) v[i] = V[i][c];
    std::vector<double> Av = mat_vec(A, v);
    for (size_t i = 0; i < 3; ++i) Av[i] -= lambda[c] * v[i];
    const double res = norm(Av);
    trace_sum += lambda[c];
    std::cout << "  lambda[" << k << "] = " << lambda[c] << "  v = (" << v[0] << ", " << v[1]
              << ", " << v[2] << ")  ||A v - lambda v|| = " << res << "\n";
    out << "eigen," << k << "," << lambda[c] << "," << res << "\n";
    if (res > 1e-9 || std::fabs(norm(v) - 1.0) > 1e-9) ++failures;
  }
  std::cout << "  sum of eigenvalues = " << trace_sum << " (trace check)\n";
  if (std::fabs(trace_sum - 9.0) > 1e-9) ++failures;

  // ---- thin SVD -----------------------------------------------------------
  const Matrix B{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  Matrix U, Vs;
  std::vector<double> S;
  dsts::svd_thin(B, U, S, Vs);
  std::cout << "\nB (4x2) singular values: " << S[0] << ", " << S[1] << "\n";
  const double err_full = frobenius_diff(B, reconstruct(U, S, Vs, 2));
  const double err_rank1 = frobenius_diff(B, reconstruct(U, S, Vs, 1));
  std::cout << "  ||B - U S V^T||_F        = " << err_full << "\n";
  std::cout << "  rank-1 truncation error = " << err_rank1 << " (should equal sigma_2 = " << S[1]
            << ")\n";
  // Orthonormal columns of U: u_i . u_j = delta_ij.
  double max_ortho = 0.0;
  for (size_t a = 0; a < 2; ++a)
    for (size_t b = 0; b < 2; ++b) {
      double dot = 0.0;
      for (size_t r = 0; r < 4; ++r) dot += U[r][a] * U[r][b];
      max_ortho = std::max(max_ortho, std::fabs(dot - (a == b ? 1.0 : 0.0)));
    }
  std::cout << "  max |U^T U - I|         = " << max_ortho << "\n";
  for (size_t k = 0; k < 2; ++k) out << "singular," << k << "," << S[k] << "," << err_full << "\n";
  out << "rank1_error,0," << err_rank1 << "," << max_ortho << "\n";
  if (err_full > 1e-9 || S[0] < S[1] || std::fabs(err_rank1 - S[1]) > 1e-9 || max_ortho > 1e-9)
    ++failures;

  std::cout << "\nwrote " RUN_OUTPUT_DIR "/worked_example.csv\n";
  std::cout << (failures == 0 ? "all checks passed\n" : "CHECKS FAILED\n");
  return failures == 0 ? 0 : 1;
}
