// 04_error_handling/03_noexcept.cpp
// noexcept: promising the compiler that a function cannot throw.
//
// WHAT YOU LEARN:
//   - noexcept after a signature is a PROMISE: "I will never throw".
//   - static_assert(noexcept(func(args))) verifies it at compile time.
//   - Useful for functions like swap and move - the STL only uses
//     move-aware containers that cannot fail.
//   - Working with doubles that CAN fail? Return a fallback instead of
//     throwing - then noexcept stays honest.
//
// EXPECTED OUTPUT:
//   before swap: a=3 b=7 ; after swap: a=7 b=3
//   safe_divide(1, 0) returned the fallback 0 (no exception possible)
//   compile-time check ok: my_swap is noexcept
//   compile-time check ok: safe_divide is noexcept
//   compile-time check NOT ok: sqrt_guard is not noexcept
//     (see static_assert below - it is commented out on purpose)

#include <iostream>

// --- a function that genuinely cannot fail ---
void my_swap(int& x, int& y) noexcept {
    int temp = x;
    x = y;
    y = temp;
}

// --- a function that could fail but chooses a fallback value ---
// Because every path returns (never throws), noexcept is honest.
double safe_divide(double a, double b) noexcept {
    return (b == 0.0) ? 0.0 : a / b;
}

// --- a function that COULD throw (compare with static_assert below) ---
double sqrt_guard(double x) {
    if (x < 0.0) {
        throw 1;   // any throw makes sqrt_guard NOT noexcept
    }
    return x;
}

int main() {
    int a = 3;
    int b = 7;
    my_swap(a, b);
    std::cout << "before swap: a=3 b=7 ; after swap: a=" << a << " b=" << b
              << "\n";

    std::cout << "safe_divide(1, 0) returned the fallback "
              << safe_divide(1.0, 0.0) << " (no exception possible)\n";

    // static_assert fails to compile if the promise is broken.
    static_assert(noexcept(my_swap(a, b)), "my_swap must never throw");
    static_assert(noexcept(safe_divide(1.0, 0.0)),
                  "safe_divide must never throw");
    std::cout << "compile-time check ok: my_swap is noexcept\n";
    std::cout << "compile-time check ok: safe_divide is noexcept\n";

    // Uncomment the next line to SEE it refuse to compile:
    //   static_assert(noexcept(sqrt_guard(1.0)),
    //                 "sqrt_guard is NOT noexcept - it can throw");
    std::cout << "compile-time check NOT ok: sqrt_guard is not noexcept\n";
    std::cout << "    (see static_assert below - it is commented out "
                 "on purpose)\n";

    return 0;
}