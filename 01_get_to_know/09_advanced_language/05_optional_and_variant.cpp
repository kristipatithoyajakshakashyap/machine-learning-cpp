// 09_advanced_language/05_optional_and_variant.cpp
// std::optional and std::variant: honest "maybe" and "one of" types.
//
// WHAT YOU LEARN:
//   - std::optional<T> either holds a T or holds NOTHING. The textbook
//     case: a division that can fail without exceptions.
//   - has_value() / value_or(fallback) make the "no value" path visible.
//   - std::variant<A,B,C> is ONE value, currently of type A, B, or C.
//   - std::holds_alternative<T>(v) asks what it currently is;
//     std::get<T>(v) pulls the value out (throws if it is the wrong type).
//
// EXPECTED OUTPUT:
//   safe_divide(10, 2)     -> 5 (has_value true)
//   safe_divide(1, 0)      -> no value (division by zero)
//   safe_divide(9, 0)      -> fallback 0 via value_or
//   variant current value  -> 3.5 (a double, int is not held)
//   variant now            -> turbo (a string)

#include <iostream>
#include <optional>
#include <string>
#include <variant>

std::optional<double> safe_divide(double a, double b) {
    if (b == 0.0) {
        return std::nullopt;             // "no answer", no exception
    }
    return a / b;
}

int main() {
    auto o1 = safe_divide(10, 2);
    std::cout << "safe_divide(10, 2)     -> " << *o1
              << " (has_value " << std::boolalpha << o1.has_value() << ")\n";

    auto o2 = safe_divide(1, 0);
    if (o2) {
        std::cout << "safe_divide(1, 0)      -> " << *o2 << "\n";
    } else {
        std::cout << "safe_divide(1, 0)      -> no value (division by "
                     "zero)\n";
    }

    double fallback = safe_divide(9, 0).value_or(0.0);
    std::cout << "safe_divide(9, 0)      -> fallback " << fallback
              << " via value_or\n";

    // --- variant: exactly one of these types is stored at a time ---
    std::variant<int, double, std::string> bag = 42;   // starts as int
    bag = 3.5;                                          // now a double
    if (std::holds_alternative<double>(bag)) {
        std::cout << "variant current value  -> " << std::get<double>(bag)
                  << " (a double, int is not held)\n";
    }

    bag = std::string("turbo");
    std::cout << "variant now            -> " << std::get<std::string>(bag)
              << " (a string)\n";
    return 0;
}