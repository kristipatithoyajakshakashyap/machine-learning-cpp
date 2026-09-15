// 01_basics/02_operators.cpp
// Operators: arithmetic, comparison, logical, bitwise, ternary.
//
// WHAT YOU LEARN:
//   - Arithmetic: + - * / % (modulo = remainder).
//   - Comparison: == != < > <= >=  (yield bool).
//   - Logical:    && (and), || (or), ! (not).
//   - Bitwise:    & | ^ ~ << >> (low-level flags and bit tricks).
//   - Ternary:    condition ? if_true : if_false.
//
// EXPECTED OUTPUT:
//   sum 15, diff 5, product 50, quotient 2, remainder 0
//   7 / 2 = 3 (int), 7.0 / 2 = 3.5 (double)
//   3 < 7 is true ; 3 == 7 is false
//   (4 > 3) && (10 < 20) is true
//   (4 > 3) || (10 > 20) is true
//   !true is false
//   5 &  3 = 1    (binary 101 & 011 = 001)
//   5 |  3 = 7    (binary 101 | 011 = 111)
//   5 ^  3 = 6    (binary 101 ^ 011 = 110)
//   ~0   = -1
//   1 << 4 = 16   (shift left = multiply by 2^4)
//   grade check: 73 is a passing grade
//   max of 12 and 8 is 12

#include <iostream>

int main() {
    int x = 10;
    int y = 5;

    // --- Arithmetic ---
    std::cout << "sum " << (x + y) << ", diff " << (x - y)
              << ", product " << (x * y) << ", quotient " << (x / y)
              << ", remainder " << (x % y) << "\n";

    // Division behavior differs for int vs double
    std::cout << "7 / 2 = " << (7 / 2) << " (int), "
              << "7.0 / 2 = " << (7.0 / 2) << " (double)\n";

    // --- Comparison ---
    bool less = (3 < 7);
    bool equal = (3 == 7);
    std::cout << std::boolalpha;  // print true/false instead of 1/0
    std::cout << "3 < 7 is " << less << " ; 3 == 7 is " << equal << "\n";

    // --- Logical ---
    bool both = (4 > 3) && (10 < 20);
    bool either = (4 > 3) || (10 > 20);
    std::cout << "(4 > 3) && (10 < 20) is " << both << "\n";
    std::cout << "(4 > 3) || (10 > 20) is " << either << "\n";
    std::cout << "!true is " << !true << "\n";

    // --- Bitwise (think: flags in libraries, hashing) ---
    std::cout << "5 &  3 = " << (5 & 3) << "    (binary 101 & 011 = 001)\n";
    std::cout << "5 |  3 = " << (5 | 3) << "    (binary 101 | 011 = 111)\n";
    std::cout << "5 ^  3 = " << (5 ^ 3) << "    (binary 101 ^ 011 = 110)\n";
    std::cout << "~0   = " << (~0) << "\n";
    std::cout << "1 << 4 = " << (1 << 4) << "   (shift left = multiply by 2^4)\n";

    // --- Ternary ---
    int grade = 73;
    std::cout << "grade check: " << grade << " is a "
              << (grade >= 60 ? "passing" : "failing") << " grade\n";
    int max = (12 > 8) ? 12 : 8;
    std::cout << "max of 12 and 8 is " << max << "\n";

    return 0;
}