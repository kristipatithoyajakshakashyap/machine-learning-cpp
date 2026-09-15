// 01_basics/05_functions.cpp
// Functions: reusable pieces of code.
//
// WHAT YOU LEARN:
//   - A function has a NAME, a RETURN TYPE, parameters, and a body.
//   - Parameters are passed BY VALUE: the function gets a copy.
//   - Default arguments let callers leave out trailing arguments.
//   - Overloading: the same name can have different parameter lists.
//   - return exits the function immediately.
//
// EXPECTED OUTPUT:
//   add(3, 4) = 7
//   greet(Alice) = Hello, Alice!
//   multiply(2.5, 4.0) = 10 ; multiply(6, 4) = 24
//   volume(2, 3, 4) = 24 ; with default height = 18
//   weather(32) = hot ; weather(22) = warm ; weather(5) = cold
//   is_even(10) = true ; is_even(7) = false

#include <iostream>
#include <string>

// --- Simple function: takes two ints, returns the sum ---
int add(int a, int b) {
    return a + b;
}

// --- Function with a string parameter ---
std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

// --- Overloading: int version AND double version ---
int    multiply(int a, int b)    { return a * b; }
double multiply(double a, double b) { return a * b; }

// --- Default argument: height is optional ---
int volume(int length, int width, int height = 3) {
    return length * width * height;
}

// --- Early return: decide the weather from a temperature ---
std::string weather(int temp) {
    if (temp >= 30) return "hot";
    if (temp >= 20) return "warm";
    return "cold";
}

// --- Returns a bool ---
bool is_even(int x) {
    return x % 2 == 0;
}

int main() {
    std::cout << "add(3, 4) = " << add(3, 4) << "\n";
    std::cout << "greet(Alice) = " << greet("Alice") << "\n";
    std::cout << "multiply(2.5, 4.0) = " << multiply(2.5, 4.0)
              << " ; multiply(6, 4) = " << multiply(6, 4) << "\n";
    std::cout << "volume(2, 3, 4) = " << volume(2, 3, 4)
              << " ; with default height = " << volume(2, 3) << "\n";
    std::cout << "weather(32) = " << weather(32) << " ; weather(22) = "
              << weather(22) << " ; weather(5) = " << weather(5) << "\n";
    std::cout << "is_even(10) = " << std::boolalpha << is_even(10)
              << " ; is_even(7) = " << is_even(7) << "\n";

    return 0;
}