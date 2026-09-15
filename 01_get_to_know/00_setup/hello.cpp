// 00_setup/hello.cpp
// Your very first C++ program.
//
// WHAT YOU LEARN:
//   - The minimal structure of a C++ program.
//   - #include <iostream> : brings in console input/output.
//   - int main() : the entry point every program must have;
//     it returns an exit code (0 = success).
//   - std::cout : prints text to the console (character output).
//
// EXPECTED OUTPUT:
//   Hello, Machine Learning world!
//   We are using C++ version:
//   <your compiler's __cplusplus value>
//
// NOTE: __cplusplus == 201703L means C++17, 202002L means C++20,
//       202302L means C++23. We code against C++17.

#include <iostream>

int main() {
    // std::endl prints a newline, exactly like '\n' but also flushes.
    std::cout << "Hello, Machine Learning world!" << std::endl;

    std::cout << "We are using C++ version: " << __cplusplus << std::endl;

    // Exit code 0 tells the operating system: everything went fine.
    return 0;
}