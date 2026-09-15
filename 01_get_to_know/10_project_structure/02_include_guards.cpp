// 10_project_structure/02_include_guards.cpp
// Include guards: why every header starts with #ifndef / #define / #endif.
//
// WHAT YOU LEARN:
//   - A header should be SAFE to include many times. The trio
//       #ifndef HELLO_UTILS_HPP
//       #define HELLO_UTILS_HPP
//       ...declarations...
//       #endif
//     makes any SECOND include a no-op: the macro is already defined, so
//     the whole body is skipped. Without the guard the compiler would see
//     the declarations twice and fail with redefinition errors.
//   - Modern C++ also offers `#pragma once`, but the #ifndef pattern is
//     the one that works everywhere and is used throughout this course.

// EXPECTED OUTPUT:
//   hello_utils.hpp included twice in this one file
//   the include guard skipped the second copy, no error
//   greetings::square(9) -> 81

#include "hello_utils.hpp"
#include "hello_utils.hpp"   // second copy is skipped by the guard

#include <iostream>

int main() {
    std::cout << "hello_utils.hpp included twice in this one file\n";
    std::cout << "the include guard skipped the second copy, no error\n";
    std::cout << "greetings::square(9) -> " << greetings::square(9)
              << "\n";
    return 0;
}