// 03_outliers.cpp
// Lesson: Tukey's IQR rule flags a value as an outlier when it lies below
// q1 - 1.5*iqr or above q3 + 1.5*iqr. Outliers are inspected, then removed
// only when they are measurement errors - never blindly.
// Equivalent: a box plot's whiskers in matplotlib/seaborn.
//
// Data: tips.csv (real data; 9 total_bill and 9 tip outliers).
//
// EXPECTED OUTPUT:
// ## total_bill
// q1    = 13.3475
// q3    = 24.1275
// iqr   = 10.78
// fence = [-2.8225, 40.2975]
// outlier count = 9 of 244 rows
// outlier bills: 48.27, 44.3, 41.19, 48.17, 50.81, 45.35, 40.55, 43.11, 48.33
//
// ## tip
// fence = [-0.34375, 5.90625]
// outlier count = 9 of 244 rows
// largest tip   = 10
//
// ## effect on the mean
// mean total_bill (all 244)  = 19.78594
// mean total_bill (inliers)  = 18.79868
// outlier bills            = 9

#include <iostream>
#include <vector>

#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill = tips.series("total_bill");

  const auto [lo, hi] = dsts::iqr_fences(bill);
  std::cout << "## total_bill\n"
            << "q1    = " << dsts::fmt(bill.quantile(0.25)) << "\n"
            << "q3    = " << dsts::fmt(bill.quantile(0.75)) << "\n"
            << "iqr   = "
            << dsts::fmt(bill.quantile(0.75) - bill.quantile(0.25)) << "\n"
            << "fence = [" << dsts::fmt(lo) << ", " << dsts::fmt(hi) << "]\n";

  const std::vector<bool> big = dsts::outlier_mask(bill);
  size_t n = 0;
  for (bool d : big) {
    if (d) ++n;
  }
  std::cout << "outlier count = " << n << " of " << tips.rows() << " rows\n";
  std::cout << "outlier bills:";
  bool first = true;
  for (size_t i = 0; i < big.size(); ++i) {
    if (!big[i]) continue;
    std::cout << (first ? " " : ", ") << dsts::to_text(bill.at(i));
    first = false;
  }
  std::cout << "\n\n";

  // Same rule for the tip column.
  const dsts::Series tip = tips.series("tip");
  const std::vector<bool> big_tip = dsts::outlier_mask(tip);
  size_t nt = 0;
  for (bool d : big_tip) {
    if (d) ++nt;
  }
  std::cout << "## tip\n"
            << "fence = [" << dsts::fmt(dsts::iqr_fences(tip).first) << ", "
            << dsts::fmt(dsts::iqr_fences(tip).second) << "]\n"
            << "outlier count = " << nt << " of " << tips.rows() << " rows\n"
            << "largest tip   = " << dsts::fmt(tip.max()) << "\n\n";

  // Removing outliers shifts the mean downward - all 9 bills are HIGH ones.
  std::vector<bool> inliers(big.size());
  for (size_t i = 0; i < big.size(); ++i) inliers[i] = !big[i];
  std::cout << "## effect on the mean\n"
            << "mean total_bill (all 244)  = " << dsts::fmt(bill.mean()) << "\n"
            << "mean total_bill (inliers)  = "
            << dsts::fmt(tips.filter(inliers).series("total_bill").mean()) << "\n"
            << "outlier bills            = " << n << "\n";

  return 0;
}