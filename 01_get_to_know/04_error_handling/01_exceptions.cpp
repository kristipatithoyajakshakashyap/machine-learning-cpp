// 04_error_handling/01_exceptions.cpp
// Exceptions: try, throw, catch.
//
// WHAT YOU LEARN:
//   - throw ... hands an error to whoever is ready to catch it.
//   - try { } catch (const SomeType& e) { } handles errors by TYPE.
//   - e.what() carries a human-readable message.
//   - A function can throw standard types like std::runtime_error.
//   - STACK UNWINDING: when an exception flies out of a block, the local
//     objects inside it are still cleaned up (destructors still run).
//
// EXPECTED OUTPUT:
//   divide(10,2) = 5
//   caught runtime_error: cannot divide by zero
//   grade 2 = 88
//   grade 7 = caught out_of_range: grade index out of bounds
//   stack unwinding demo:
//     cleanup() ran for notes.txt even during a throw
//     caught: boom
//     (the FileGuard printed BEFORE the catch - cleanup always ran)

#include <iostream>
#include <stdexcept>
#include <string>

// --- a function that can fail with throw ---
double divide(double a, double b) {
    if (b == 0.0) {
        throw std::runtime_error("cannot divide by zero");
    }
    return a / b;
}

// --- a function that checks an index and throws on a bad one ---
int grade_at(const int grades[], int count, int index) {
    if (index < 0 || index >= count) {
        throw std::out_of_range("grade index out of bounds");
    }
    return grades[index];
}

// --- a tiny object that reports its cleaning-up ---
struct FileGuard {
    explicit FileGuard(const char* badge) : label(badge) {}
    ~FileGuard() {
        std::cout << "    cleanup() ran for " << label
                  << " even during a throw\n";
    }
    const char* label;
};

int main() {
    // 1) throw + catch
    try {
        double r = divide(10.0, 2.0);
        std::cout << "divide(10,2) = " << r << "\n";
        divide(10.0, 0.0);              // this call throws
    } catch (const std::runtime_error& e) {
        std::cout << "caught runtime_error: " << e.what() << "\n";
    }

    // 2) out_of_range from a helper function
    int grades[5] = {90, 85, 88, 92, 79};
    try {
        std::cout << "grade 2 = " << grade_at(grades, 5, 2) << "\n";
        std::cout << "grade 7 = " << grade_at(grades, 5, 7) << "\n";
    } catch (const std::out_of_range& e) {
        std::cout << "caught out_of_range: " << e.what() << "\n";
    }

    // 3) stack unwinding: locals are cleaned up during the escape
    std::cout << "stack unwinding demo:\n";
    try {
        FileGuard guard("notes.txt");   // alive inside the try
        throw std::runtime_error("boom");
    } catch (const std::runtime_error& e) {
        std::cout << "    caught: " << e.what() << "\n";
    }
    std::cout << "    (the FileGuard printed BEFORE the catch - "
                 "cleanup always ran)\n";

    return 0;
}