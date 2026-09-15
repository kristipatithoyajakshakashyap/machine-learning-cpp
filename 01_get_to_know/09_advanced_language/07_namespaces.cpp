// 09_advanced_language/07_namespaces.cpp
// Namespaces: name packages that keep large codebases collision-free.
//
// WHAT YOU LEARN:
//   - namespace math { ... } groups related functions and constants.
//   - The SAME name can exist in two namespaces without clashing; call
//     them fully qualified (math::square) and nobody gets confused.
//   - using math::square; pulls in ONE name; using namespace shortcuts the
//     whole group (used sparingly, e.g. namespace fs = std::filesystem).
//
// EXPECTED OUTPUT:
//   alpha::title() = team alpha
//   beta::title()  = team beta
//   math::square(4) = 16
//   math::average({2, 4, 6}) = 4
//   using math::square -> square(3) = 9

#include <iostream>
#include <string>
#include <vector>

namespace alpha {
const std::string title() { return "team alpha"; }
}

namespace beta {
const std::string title() { return "team beta"; }
}

namespace math {
double square(double x) { return x * x; }

double average(const std::vector<double>& xs) {
    double sum = 0.0;
    for (double x : xs) {
        sum += x;
    }
    return sum / static_cast<double>(xs.size());
}
}  // namespace math

int main() {
    std::cout << "alpha::title() = " << alpha::title() << "\n";
    std::cout << "beta::title()  = " << beta::title() << "\n";

    std::cout << "math::square(4) = " << math::square(4) << "\n";

    std::vector<double> nums{2, 4, 6};
    std::cout << "math::average({2, 4, 6}) = "
              << math::average(nums) << "\n";

    using math::square;                    // pull just ONE name in
    std::cout << "using math::square -> square(3) = " << square(3)
              << "\n";
    return 0;
}