#ifndef HELPER_MATH_MATRIX_HPP
#define HELPER_MATH_MATRIX_HPP

// ===========================================================================
// helper/math/matrix.hpp  (+ the matching helper/math/matrix.cpp)
// ---------------------------------------------------------------------------
// Dense matrix helpers shared by EVERY model in the course (and by
// helper/data/datasets.cpp, helper/math/optim.cpp, helper/math/metrics.cpp
// and helper/eval/cross_validation.cpp).
//
// Rows are observations, columns are features, exactly the layout sklearn /
// numpy uses for a 2-D array.  Every function is deterministic and throws
// std::runtime_error on dimension mismatch rather than silently corrupting
// memory.
// ===========================================================================

#include <cstddef>
#include <string>
#include <vector>

namespace ml {

// A matrix is a vector of rows; each row is a vector of doubles.
using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

// Allocation helpers.
Mat make_mat(size_t rows, size_t cols, double init = 0.0);
Vec make_vec(size_t n, double init = 0.0);

// Core linear algebra.
Mat transpose(const Mat& A);             // T[j][i] = A[i][j]
Mat matmul(const Mat& A, const Mat& B);  // (m x k)(k x n)
Vec matvec(const Mat& A, const Vec& x);  // (m x k)(k)
double dot(const Vec& a, const Vec& b);  // scalar product
Mat identity(size_t n);
Vec linspace(double lo, double hi, size_t n);  // n evenly spaced points

// Solve A x = b by Gaussian elimination with partial pivoting (deterministic).
// Throws std::runtime_error when A is numerically singular.
Vec solve_linear(const Mat& A, const Vec& b);

// Column statistics helpers used for pre-processing.
Vec col_mean(const Mat& A);                       // mean of each column
Vec col_std(const Mat& A, bool population = true);  // std (N or N-1 denom)

// z-score standardise: (X - mean) / std per column.  std==0 columns stay 0.
// The mean/std used are optionally returned so the same transform can be
// reapplied to held-out data.
Mat standardize(const Mat& A, Vec* out_mean = nullptr, Vec* out_std = nullptr);

// Pretty printers for lesson output.
std::string print_mat(const Mat& A, size_t max_rows = 6, size_t max_cols = 6,
                      int precision = 7);
std::string print_vec(const Vec& v, size_t max_items = 10,
                      const std::string& label = "");

}  // namespace ml

#endif  // ML_MAT_HPP