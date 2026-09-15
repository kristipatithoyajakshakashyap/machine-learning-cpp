// 09_advanced_language/03_auto_and_decltype.cpp
// auto and decltype: let the compiler name types for you.
//
// WHAT YOU LEARN:
//   - auto deduces the type from the initializer on the RIGHT side.
//   - decltype(expression) names the type OF an expression without making
//     a value — great for "a variable of exactly this type".
//   - const auto& copies nothing and reads anything.
//
// EXPECTED OUTPUT:
//   auto n  = 42              -> int, prints 42
//   auto pi = 3.14159         -> double
//   auto name = string        -> std::string
//   auto v = vector{1,2,3}    -> std::vector<int>, elements 1 2 3
//   decltype(pi) second       -> a double again, prints 2.5
//   decltype(v) copy          -> vector<int> copy, size 3

#include <iostream>
#include <string>
#include <vector>

int main() {
    auto n = 42;                        // int
    std::cout << "auto n  = 42              -> int, prints " << n << "\n";

    auto pi = 3.14159;                  // double
    std::cout << "auto pi = 3.14159         -> double, prints " << pi
              << "\n";

    auto name = std::string("lambda?"); // std::string
    std::cout << "auto name = string        -> std::string, prints '"
              << name << "'\n";

    auto v = std::vector<int>{1, 2, 3}; // std::vector<int>
    std::cout << "auto v = vector{1,2,3}    -> std::vector<int>, elements ";
    for (int x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    decltype(pi) second = 2.5;          // same type as pi -> double
    std::cout << "decltype(pi) second       -> a double again, prints "
              << second << "\n";

    decltype(v) copy = v;               // exact type of v -> vector<int>
    std::cout << "decltype(v) copy          -> vector<int> copy, size "
              << copy.size() << "\n";

    const auto& view = v;               // read-only window, nothing copied
    std::cout << "const auto& view          -> read-only, first is "
              << view.front() << "\n";
    return 0;
}