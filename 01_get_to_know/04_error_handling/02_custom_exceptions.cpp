// 04_error_handling/02_custom_exceptions.cpp
// Custom exceptions: your own error type with extra information.
//
// WHAT YOU LEARN:
//   - Make your class derive from std::runtime_error (or std::exception).
//   - You can carry EXTRA data (which variable, what bad value).
//   - Catch order matters: catch the most specific type FIRST.
//   - Catching the base class (std::exception) catches every derived type.
//
// EXPECTED OUTPUT:
//   sqrt(9) = 3
//   caught custom error: negative value for sqrt input
//     variable: sqrt input, bad value: -4
//   a base-class catch also got it: negative value for sqrt input
//   continue normally after the exception

#include <cmath>       // std::sqrt
#include <iostream>
#include <stdexcept>   // std::runtime_error
#include <string>

// --- our own exception: a negative value error with details ---
class NegativeValueError : public std::runtime_error {
public:
    NegativeValueError(const std::string& variable, double bad_value)
        : std::runtime_error("negative value for " + variable),
          variable_(variable),
          bad_value_(bad_value) {}

    const std::string& variable() const { return variable_; }
    double            bad_value() const { return bad_value_; }

private:
    std::string variable_;
    double      bad_value_;
};

// --- a function that throws our custom type ---
double safe_sqrt(double x) {
    if (x < 0.0) {
        throw NegativeValueError("sqrt input", x);
    }
    return std::sqrt(x);
}

int main() {
    // catch the MOST specific type first (NegativeValueError)
    try {
        std::cout << "sqrt(9) = " << safe_sqrt(9.0) << "\n";
        safe_sqrt(-4.0);                 // throws our custom error
    } catch (const NegativeValueError& e) {
        std::cout << "caught custom error: " << e.what() << "\n";
        std::cout << "    variable: " << e.variable() << ", bad value: "
                  << e.bad_value() << "\n";
    }

    // a base-class catch catches ALL derived errors
    try {
        safe_sqrt(-4.0);
    } catch (const std::exception& e) {  // catches NegativeValueError too
        std::cout << "a base-class catch also got it: " << e.what() << "\n";
    }

    std::cout << "continue normally after the exception\n";
    return 0;
}