// 10_project_structure/03_matrix_library.cpp
// A header-only Matrix class, consumed like any library.
//
// WHAT YOU LEARN:
//   - Matrix.hpp is a HEADER-ONLY library: all methods are defined inside
//     the header, so using it is just #include "Matrix.hpp".
//   - Matrix is stored in one flat std::vector (row 0 first, then row 1):
//     data[r * cols + c] is the element at (row r, column c). Fast for
//     the CPU because everything sits in one block of memory.
//   - operator() gives the (r, c) syntax you know from math; the const
//     overload handles read-only matrices.
//
// EXPECTED OUTPUT:
//   Matrix(2, 3) -> 2 rows x 3 cols
//   contents:
//   1 2 3
//   4 5 6
//   element sum = 21
//   m(0,0) = 9 -> first element now 9

#include "Matrix.hpp"

#include <iostream>

int main() {
    Matrix m(2, 3);

    // fill the matrix with 1..6
    double value = 1.0;
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            m(r, c) = value;
            value += 1.0;
        }
    }

    std::cout << "Matrix(2, 3) -> " << m.rows() << " rows x " << m.cols()
              << " cols\n";
    std::cout << "contents:\n";

    double sum = 0.0;
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            std::cout << m(r, c) << " ";
            sum += m(r, c);
        }
        std::cout << "\n";
    }
    std::cout << "element sum = " << sum << "\n";

    // write through the non-const operator()
    m(0, 0) = 9.0;
    std::cout << "m(0,0) = 9 -> first element now " << m(0, 0) << "\n";
    return 0;
}