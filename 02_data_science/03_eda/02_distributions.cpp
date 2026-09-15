// 02_distributions.cpp
// Lesson: a histogram shows where the data concentrates. histogram() chops
// the range into equal-width bins and counts how many values land in each;
// dividing by the total gives proportions. Missing cells are skipped.
// Equivalent: numpy.histogram, plt.hist.
//
// Data: iris.csv and tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## petal_length, 10 equal-width bins
// range = [1, 6.9]
// [1, 1.59)  count  37  0.2466667
// [1.59, 2.18)  count  13  0.08666667
// [2.18, 2.77)  count   0  0
// [2.77, 3.36)  count   3  0.02
// [3.36, 3.95)  count   8  0.05333333
// [3.95, 4.54)  count  26  0.1733333
// [4.54, 5.13)  count  29  0.1933333
// [5.13, 5.72)  count  18  0.12
// [5.72, 6.31)  count  11  0.07333333
// [6.31, 6.9)  count   5  0.03333333
// total in bins = 150
//
// ## total_bill, 8 bins (right-skewed)
// [3.07, 9.0375)  count  12
// [9.0375, 15.005)  count  68
// [15.005, 20.9725)  count  81
// [20.9725, 26.94)  count  38
// [26.94, 32.9075)  count  24
// [32.9075, 38.875)  count  10
// [38.875, 44.8425)  count   6
// [44.8425, 50.81)  count   5
//
// the bulk sits below $30; the tail extends to $50.81

#include <iomanip>
#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::Series petal = iris.series("petal_length");
  const dsts::Histogram h = dsts::histogram(petal, 10);

  std::cout << "## petal_length, 10 equal-width bins\n";
  std::cout << "range = [" << dsts::fmt(petal.min()) << ", "
            << dsts::fmt(petal.max()) << "]\n";
  for (size_t i = 0; i < h.counts.size(); ++i) {
    const double frac = static_cast<double>(h.counts[i]) /
                        static_cast<double>(petal.count());
    std::cout << "[" << dsts::fmt(h.edges[i]) << ", " << dsts::fmt(h.edges[i + 1])
              << ")  count " << std::setw(3) << h.counts[i] << "  " << dsts::fmt(frac)
              << "\n";
  }
  std::cout << "total in bins = " << petal.count() << "\n\n";

  // A normal-looking distribution: tips total_bill.
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Histogram b = dsts::histogram(tips.series("total_bill"), 8);
  std::cout << "## total_bill, 8 bins (right-skewed)\n";
  for (size_t i = 0; i < b.counts.size(); ++i) {
    std::cout << "[" << dsts::fmt(b.edges[i]) << ", " << dsts::fmt(b.edges[i + 1])
              << ")  count " << std::setw(3) << b.counts[i] << "\n";
  }
  std::cout << "\nthe bulk sits below $30; the tail extends to $50.81\n";

  return 0;
}