// 11_capstone/03_dataset_normalizer.cpp
// Normalization: rescaling a column so every value lands in [0, 1].
//
// WHAT YOU LEARN:
//   - Columns are the thing you normalize, one at a time.
//   - MIN-MAX normalization:  value' = (value - min) / (max - min).
//     The smallest value becomes 0, the largest becomes 1.
//   - This keeps the SHAPE of the data but removes the scale, so a
//     column measured in centimeters and one measured in kilograms can
//     be compared on equal footing.
//   - Guard: if max == min the formula divides by zero — check first.
//
// EXPECTED OUTPUT:
//   heights          : 170 182 158 175   (min 158, max 182)
//   normalized       : 0.5 1 0 0.708
//   min maps to 0, max maps to 1
//   original stayed put: 170 182 158 175

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

double min_of(const std::vector<double>& xs) {
    if (xs.empty()) throw std::invalid_argument("minimum needs a nonempty vector");
    return *std::min_element(xs.begin(), xs.end());
}

double max_of(const std::vector<double>& xs) {
    if (xs.empty()) throw std::invalid_argument("maximum needs a nonempty vector");
    return *std::max_element(xs.begin(), xs.end());
}

// Map each value of xs into [0, 1] using the column's own range.
std::vector<double> min_max_normalize(const std::vector<double>& xs) {
    if (xs.empty()) return {};
    for (double x : xs) if (!std::isfinite(x)) throw std::invalid_argument("normalization requires finite values");
    std::vector<double> out(xs.size());
    double lo = min_of(xs);
    double hi = max_of(xs);
    double span = hi - lo;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        out[i] = span == 0.0 ? 0.0 : (xs[i] - lo) / span;
    }
    return out;
}

void print(const char* label, const std::vector<double>& v) {
    std::cout << label << ": ";
    for (double x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    const std::vector<double> heights{170, 182, 158, 175};

    double lo = min_of(heights);
    double hi = max_of(heights);
    std::cout << "heights          : ";
    for (double h : heights) {
        std::cout << h << " ";
    }
    std::cout << "  (min " << lo << ", max " << hi << ")\n";

    std::cout << std::setprecision(3);
    std::vector<double> scaled = min_max_normalize(heights);
    print("normalized       ", scaled);
    std::cout << "min maps to 0, max maps to 1\n";
    std::cout << "original stayed put: ";
    for (double h : heights) {
        std::cout << h << " ";
    }
    std::cout << "\n";
    return 0;
}
