// 10_project_structure/01_program_layout.cpp
// Program layout: how a real project is divided into files.
//
// WHAT YOU LEARN:
//   - HEADERS (.hpp) carry declarations + doc comments; SOURCES (.cpp)
//     carry the definitions.
//   - Each .cpp is compiled separately; the static library bundles
//     object code; executables LINK the library to use its functions.
//   - This file only #includes "hello_utils.hpp" — it never sees the
//     implementation in src/hello_utils.cpp.

// EXPECTED OUTPUT:
//   real programs are split into headers + sources
//   greetings::cheerful("ada") -> hi ada!
//   greetings::square(12)      -> 144
//   hello_utils.cpp is compiled into a static library
//   conc_program_layout.exe links that library

#include "hello_utils.hpp"

#include <iostream>

int main() {
    std::cout << "real programs are split into headers + sources\n";
    std::cout << "greetings::cheerful(\"ada\") -> "
              << greetings::cheerful("ada") << "\n";
    std::cout << "greetings::square(12)      -> "
              << greetings::square(12) << "\n";
    std::cout << "hello_utils.cpp is compiled into a static library\n";
    std::cout << "conc_program_layout.exe links that library\n";
    return 0;
}