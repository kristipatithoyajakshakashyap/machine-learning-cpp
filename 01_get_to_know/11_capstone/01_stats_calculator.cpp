// 11_capstone/01_stats_calculator.cpp
// Statistics you can trust: the numbers behind every data table.
//
// WHAT YOU LEARN:
//   - MEAN    = the average (sum / count).
//   - MEDIAN  = the middle value of a SORTED list (average of the two
//     middles when the count is even). Robust to extreme values.
//   - MIN/MAX = the outer edges of the data.
//   - STD DEV = how far values typically sit from the mean; here the
//     population version spreads the squared gaps by the count.
//   - Sorting a copy is the clean way to find the median.
//
// EXPECTED OUTPUT:
//   sample: 2 4 4 4 5 5 7 9
//   sorted: 2 4 4 4 5 5 7 9
//   mean   = 5
//   median = 4.5
//   min    = 2
//   max    = 9
//   std dev = 2  (variance 4, so values sit ~2 away from the mean)

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

double mean(const std::vector<double>& xs) {
    double total = 0.0;
    for (double x : xs) {
        total += x;
    }
    return total / static_cast<double>(xs.size());
}

double median(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());          // work on a sorted copy
    std::size_t n = xs.size();
    if (n % 2 == 1) {
        return xs[n / 2];                     // odd count: the middle
    }
    return (xs[n / 2 - 1] + xs[n / 2]) / 2.0; // even: middle two
}

double population_stddev(const std::vector<double>& xs) {
    double m = mean(xs);
    double squared_gaps = 0.0;
    for (double x : xs) {
        squared_gaps += (x - m) * (x - m);
    }
    double variance = squared_gaps / static_cast<double>(xs.size());
    return std::sqrt(variance);
}

int main() {
    const std::vector<double> sample{2, 4, 4, 4, 5, 5, 7, 9};

    std::cout << std::setprecision(3);
    std::cout << "sample: ";
    for (double x : sample) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::vector<double> sorted = sample;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "sorted: ";
    for (double x : sorted) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::cout << "mean   = " << mean(sample) << "\n";
    std::cout << "median = " << median(sample) << "\n";
    std::cout << "min    = " << sorted.front() << "\n";
    std::cout << "max    = " << sorted.back() << "\n";
    std::cout << "std dev = " << population_stddev(sample)
              << "  (variance 4, so values sit ~2 away from the mean)\n";
    return 0;
}