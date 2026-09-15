#include "helper/math/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ml {
// ---------------------------------------------------------------------------
// Basic linear-algebra helpers.
//
// All functions here are free, deterministic, and exception-safe:
// dimension-mismatch errors throw std::runtime_error so lessons report a
// clear failure instead of silently reading past a vector's end.
// ---------------------------------------------------------------------------

// Allocate a rows x cols matrix (all cells `init`).
Mat make_mat(size_t rows, size_t cols, double init) {
  return Mat(rows, Vec(cols, init));
}

// Allocate a length-n vector (all cells `init`).
Vec make_vec(size_t n, double init) { return Vec(n, init); }

// Matrix transpose: T[j][i] = A[i][j].  Returns {} for an empty input so the
// result is always a valid (possibly 0 x 0) matrix.
Mat transpose(const Mat& A) {
  if (A.empty()) return {};
  Mat T(A[0].size(), Vec(A.size()));
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
  return T;
}

// Dense (m x k) @ (k x n) matrix product.  The naive triple loop is O(m*k*n),
// fine for the lesson-sized data; no BLAS, no surprises.
Mat matmul(const Mat& A, const Mat& B) {
  if (A.empty() || B.empty() || A[0].size() != B.size())
    throw std::runtime_error("matmul: dimension mismatch");
  Mat C(A.size(), Vec(B[0].size(), 0.0));
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < B[0].size(); ++j) {
      double s = 0.0;
      for (size_t k = 0; k < A[0].size(); ++k) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  return C;
}

// Matrix-vector product (m x k) @ (k).  The rows of A are shrunk to scalars
// via a dot product with x.
Vec matvec(const Mat& A, const Vec& x) {
  if (A.empty() || A[0].size() != x.size())
    throw std::runtime_error("matvec: dimension mismatch");
  Vec y(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
  return y;
}

// Scalar dot product.
double dot(const Vec& a, const Vec& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// Identity matrix of size n (1.0 on the diagonal, 0 elsewhere).
Mat identity(size_t n) {
  Mat I(n, Vec(n, 0.0));
  for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
  return I;
}

// n points spread evenly from lo to hi (inclusive).  Used for plotting
// curves and for scanning regularisation grids.
Vec linspace(double lo, double hi, size_t n) {
  Vec v(n);
  if (n == 1) {
    v[0] = lo;
    return v;
  }
  double step = (hi - lo) / static_cast<double>(n - 1);
  for (size_t i = 0; i < n; ++i) v[i] = lo + step * static_cast<double>(i);
  return v;
}

// Solve A x = b by Gaussian elimination with partial pivoting.
//
// The augmented matrix [A | b] is reduced to row-echelon form by, for each
// column: (1) choose the row with the largest |A[r][col]| as pivot (guards
// against dividing by near-zero values), (2) normalise the pivot row,
// (3) eliminate that column from every row below.  Back substitution then
// fills x from the bottom up.  Because pivoting is deterministic, the same
// input always yields the same solution (required for fixed lesson output).
Vec solve_linear(const Mat& A, const Vec& b) {
  size_t n = A.size();
  if (n == 0 || A[0].size() != n || b.size() != n)
    throw std::runtime_error("solve_linear: dimension mismatch");

  // Build the augmented matrix [A | b] as n x (n+1).
  Mat M(n, Vec(n + 1));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) M[i][j] = A[i][j];
    M[i][n] = b[i];
  }

  // Forward elimination.
  for (size_t col = 0; col < n; ++col) {
    // Partial pivoting: swap in the largest-magnitude pivot row.
    size_t pivot = col;
    double best = std::fabs(M[col][col]);
    for (size_t r = col + 1; r < n; ++r) {
      if (std::fabs(M[r][col]) > best) {
        best = std::fabs(M[r][col]);
        pivot = r;
      }
    }
    // A (numerically) singular system: no unique solution, bail out loudly.
    if (best < 1e-300) throw std::runtime_error("solve_linear: singular matrix");
    if (pivot != col) std::swap(M[col], M[pivot]);

    // Normalise the pivot row (make the diagonal entry 1).
    double factor = M[col][col];
    for (size_t j = col; j <= n; ++j) M[col][j] /= factor;

    // Eliminate column `col` from all later rows.
    for (size_t r = col + 1; r < n; ++r) {
      double f = M[r][col];
      for (size_t j = col; j <= n; ++j) M[r][j] -= f * M[col][j];
    }
  }

  // Back substitution (bottom row first).
  Vec x(n);
  for (long col = static_cast<long>(n) - 1; col >= 0; --col) {
    double s = M[col][n];
    for (size_t j = static_cast<size_t>(col) + 1; j < n; ++j)
      s -= M[col][j] * x[j];
    x[static_cast<size_t>(col)] = s;
  }
  return x;
}

// Mean of each column (length = number of columns).
Vec col_mean(const Mat& A) {
  if (A.empty()) return {};
  size_t p = A[0].size();
  Vec m(p, 0.0);
  for (const auto& row : A)
    for (size_t j = 0; j < p; ++j) m[j] += row[j];
  for (double& v : m) v /= static_cast<double>(A.size());
  return m;
}

// Standard deviation of each column.  population=true uses N as the
// denominator, population=false uses N-1 (sample std).  A single-row or
// empty-vector denominator is clamped to 1 to avoid divide-by-zero.
Vec col_std(const Mat& A, bool population) {
  if (A.empty()) return {};
  size_t p = A[0].size();
  Vec m = col_mean(A);
  Vec s(p, 0.0);
  for (const auto& row : A)
    for (size_t j = 0; j < p; ++j) s[j] += (row[j] - m[j]) * (row[j] - m[j]);
  double denom =
      static_cast<double>(A.size()) - (population ? 0.0 : 1.0);
  if (denom <= 0.0) denom = 1.0;
  for (double& v : s) v = std::sqrt(v / denom);
  return s;
}

// z-score standardisation: (X - mean) / std per column.  Zero-variance
// columns (std == 0) are left at 0 instead of producing NaN.  The mean and
// std actually used are written back through out_mean/out_std so the caller
// can apply the same transform to later data (e.g. the test set).
Mat standardize(const Mat& A, Vec* out_mean, Vec* out_std) {
  Vec m = col_mean(A);
  Vec s = col_std(A, false);  // sample std so zero-variance columns stay 0
  Mat Z = A;
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < A[0].size(); ++j)
      Z[i][j] = s[j] > 1e-15 ? (A[i][j] - m[j]) / s[j] : 0.0;
  if (out_mean) *out_mean = m;
  if (out_std) *out_std = s;
  return Z;
}

// ---------------------------------------------------------------------------
// Printers (used for lesson output; nothing here affects model behaviour).
// ---------------------------------------------------------------------------

// Compact matrix printout: up to max_rows x max_cols entries followed by the
// full shape "rows x cols" on its own line.
std::string print_mat(const Mat& A, size_t max_rows, size_t max_cols,
                      int precision) {
  if (A.empty()) return "<empty>";
  std::ostringstream o;
  o << std::fixed << std::setprecision(precision);
  size_t r_end = std::min(A.size(), max_rows);
  size_t c_end = std::min(A[0].size(), max_cols);
  for (size_t i = 0; i < r_end; ++i) {
    for (size_t j = 0; j < c_end; ++j) {
      if (j) o << " ";
      o << std::setw(12) << A[i][j];
    }
    if (A[0].size() > c_end) o << " ...";
    o << "\n";
  }
  if (A.size() > r_end) o << "...\n";
  o << A.size() << " x " << A[0].size();
  return o.str();
}

// Compact vector printout; an optional label prefixes the values, a trailing
// "..." marks truncated output for long vectors.
std::string print_vec(const Vec& v, size_t max_items, const std::string& label) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(7);
  if (!label.empty()) o << label << " ";
  size_t n = std::min(v.size(), max_items);
  for (size_t i = 0; i < n; ++i) {
    if (i) o << " ";
    o << v[i];
  }
  if (v.size() > n) o << " ...";
  return o.str();
}

}  // namespace ml