// 11_capstone/05_projects_tests.cpp
// Your own test harness: PASS/FAIL + a summary, no framework needed.
//
// WHAT YOU LEARN:
//   - A CHECK(condition, message) macro records every outcome.
//   - Doubles are compared with a TOLERANCE (exact == fails on 0.1)
//   - Tests call the same functions the real program uses, with known
//     inputs and expected results: regression proof for later modules.
//
// EXPECTED OUTPUT:
//   [PASS] mean of 8 values is 5
//   [PASS] median of even count is 4.5
//   [PASS] min-max normalize maps min to 0
//   [PASS] min-max normalize maps max to 1
//   [PASS] middle value normalizes to 0.5
//   5 checks, 0 failed

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static int checks_run = 0;
static int checks_failed = 0;

#define CHECK(condition, message)                 \
    do {                                          \
        ++checks_run;                             \
        if (condition) {                          \
            std::cout << "[PASS] " << message << "\n";   \
        } else {                                  \
            ++checks_failed;                      \
            std::cout << "[FAIL] " << message << "\n";   \
        }                                         \
    } while (0)

// The piece of logic under test (mirrors 01_stats_calculator).
double mean(const std::vector<double>& xs) {
    double total = 0.0;
    for (double x : xs) {
        total += x;
    }
    return total / static_cast<double>(xs.size());
}

double median(const std::vector<double>& xs) {
    std::vector<double> sorted = xs;
    std::sort(sorted.begin(), sorted.end());
    std::size_t n = sorted.size();
    if (n % 2 == 1) {
        return sorted[n / 2];
    }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
}

// Min-max normalization (mirrors 03_dataset_normalizer).
double normalize(double value, double lo, double hi) {
    return (value - lo) / (hi - lo);
}

int main() {
    const std::vector<double> a{2, 4, 4, 4, 5, 5, 7, 9};

    // tolerate tiny floating-point noise
    const double kTol = 1e-9;

    CHECK(std::fabs(mean(a) - 5.0) < kTol, "mean of 8 values is 5");
    CHECK(std::fabs(median(a) - 4.5) < kTol, "median of even count is 4.5");
    CHECK(normalize(158, 158, 182) == 0.0,
          "min-max normalize maps min to 0");
    CHECK(normalize(182, 158, 182) == 1.0,
          "min-max normalize maps max to 1");
    CHECK(std::fabs(normalize(170, 158, 182) - 0.5) < kTol,
          "middle value normalizes to 0.5");

    std::cout << checks_run << " checks, " << checks_failed << " failed\n";
    return checks_failed == 0 ? 0 : 1;
}