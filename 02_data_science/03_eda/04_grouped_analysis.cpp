// 04_grouped_analysis.cpp
// Lesson: grouping compares slices of the data with one another. group_mean()
// computes count/mean/sd of a numeric column inside every group of a string
// column - the pandas groupby().mean() pattern.
// Equivalent: df.groupby(...).mean().
//
// Data: tips.csv and iris.csv (real data).
//
// EXPECTED OUTPUT:
// ## mean total_bill by day
// Fri        n=19   mean=17.15158   sd=8.30266
// Sat        n=87   mean=20.44138   sd=9.480419
// Sun        n=76   mean=21.41      sd=8.832122
// Thur       n=62   mean=17.68274   sd=7.88617
//
// ## mean tip by smoker
// No         n=151  mean=2.991854   sd=1.37719
// Yes        n=93   mean=3.00871    sd=1.401468
//
// ## mean petal_length by species (iris)
// setosa       n=50   mean=1.462      sd=0.173664
// versicolor   n=50   mean=4.26       sd=0.469911
// virginica    n=50   mean=5.552      sd=0.5518947

#include <iomanip>
#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");

  std::cout << "## mean total_bill by day\n";
  for (const auto& g : dsts::group_mean(tips, "day", "total_bill")) {
    std::cout << std::left << std::setw(10) << g.group << " n=" << std::setw(3)
              << g.count << "  mean=" << std::setw(9) << dsts::fmt(g.mean)
              << "  sd=" << dsts::fmt(g.sd) << "\n";
  }

  std::cout << "\n## mean tip by smoker\n";
  for (const auto& g : dsts::group_mean(tips, "smoker", "tip")) {
    std::cout << std::left << std::setw(10) << g.group << " n=" << std::setw(3)
              << g.count << "  mean=" << std::setw(9) << dsts::fmt(g.mean)
              << "  sd=" << dsts::fmt(g.sd) << "\n";
  }

  std::cout << "\n## mean petal_length by species (iris)\n";
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  for (const auto& g : dsts::group_mean(iris, "species", "petal_length")) {
    std::cout << std::left << std::setw(12) << g.group << " n=" << std::setw(3)
              << g.count << "  mean=" << std::setw(9) << dsts::fmt(g.mean)
              << "  sd=" << dsts::fmt(g.sd) << "\n";
  }

  return 0;
}