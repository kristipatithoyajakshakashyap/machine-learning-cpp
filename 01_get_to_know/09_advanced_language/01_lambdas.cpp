// 09_advanced_language/01_lambdas.cpp
// Lambdas: functions you write inline, right where you need them.
//
// WHAT YOU LEARN:
//   - [](int x) { return x % 2 == 0; } IS a function value.
//   - [fixed] captures a variable BY VALUE (a copy is taken now).
//   - [&base] captures BY REFERENCE; later changes to base are visible.
//   - Lambdas plug directly into STL algorithms (count_if, transform, ...).
//
// EXPECTED OUTPUT:
//   count_if evens (lambda)   : 3
//   capture by value [fixed]  : add_fixed(5) = 15
//   capture by ref [&base]    : scale(3) = 300
//   ref capture sees updates  : after base=200, scale(3) = 600
//   transform via lambda      : 2 4 6 8 10 12

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> nums{1, 2, 3, 4, 5, 6};

    // Named lambda: stored in an auto variable like any function.
    auto is_even = [](int x) { return x % 2 == 0; };
    int evens = std::count_if(nums.begin(), nums.end(), is_even);
    std::cout << "count_if evens (lambda)   : " << evens << "\n";

    // Capture by value: copies 'fixed' into the lambda at creation time.
    int fixed = 10;
    auto add_fixed = [fixed](int x) { return x + fixed; };
    std::cout << "capture by value [fixed]  : add_fixed(5) = "
              << add_fixed(5) << "\n";

    // Capture by reference: the lambda looks at the LIVE variable.
    int base = 100;
    auto scale = [&base](int x) { return base * x; };
    std::cout << "capture by ref [&base]    : scale(3) = "
              << scale(3) << "\n";

    base = 200;                             // visible through the reference
    std::cout << "ref capture sees updates  : after base=200, scale(3) = "
              << scale(3) << "\n";

    // Build a new vector with the lambda mapping each element.
    std::vector<int> shifted(nums.size());
    std::transform(nums.begin(), nums.end(), shifted.begin(),
                   [](int x) { return x * 2; });
    std::cout << "transform via lambda      : ";
    for (int x : shifted) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}