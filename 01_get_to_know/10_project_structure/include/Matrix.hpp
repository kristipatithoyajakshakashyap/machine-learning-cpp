#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>   // std::size_t
#include <vector>

// A tiny row-major Matrix stored in one flat vector.
// HEADER-ONLY: every method is defined right here, so a user program only
// needs #include "Matrix.hpp" (compiled once per .cpp that uses it).
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Non-const version: returns a REFERENCE so you can write m(r, c) = v.
    double& operator()(std::size_t r, std::size_t c) {
        return data_.at(r * cols_ + c);
    }

    // Const version: read-only (data_.at returns a const value here).
    double operator()(std::size_t r, std::size_t c) const {
        return data_.at(r * cols_ + c);
    }

private:
    std::size_t          rows_;
    std::size_t          cols_;
    std::vector<double>  data_;
};

#endif  // MATRIX_HPP