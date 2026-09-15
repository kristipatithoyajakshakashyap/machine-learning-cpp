// 04_normality.cpp
// Lesson: many test assumptions start with "is this column Normal?".
// ks_normal() runs a one-sample Kolmogorov-Smirnov test against a standard
// normal (after z-standardising), returning D and the two-sided p-value.
// A large p means "no evidence against normality".
// Equivalent: scipy.stats.kstest(x, "norm") on z-scored data.
//
// Data: iris.csv, tips.csv, penguins.csv (real data).
//
// EXPECTED OUTPUT:
// tips total_bill  n=244  D=0.1187539  p=0.001833986
// tips tip  n=244  D=0.1016534  p=0.01189164
// iris sepal_length  n=150  D=0.08865361  p=0.1799016
// iris petal_length  n=150  D=0.1981541  p=1.193805e-05
// penguins flipper_length_mm  n=342  D=0.1242788  p=4.468959e-05

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

void check(const std::string& label, const dsts::Series& x) {
  const dsts::KsResult k = dsts::ks_normal(x);
  std::cout << label << "  n=" << x.count() << "  D=" << dsts::fmt(k.statistic)
            << "  p=" << dsts::fmt(k.p_value) << "\n";
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::DataFrame penguins = dsts::read_csv(DATA_DIR "/penguins.csv");

  check("tips total_bill", dsts::Series("total_bill", tips.numeric("total_bill")));
  check("tips tip", dsts::Series("tip", tips.numeric("tip")));
  check("iris sepal_length", dsts::Series("sl", iris.numeric("sepal_length")));
  check("iris petal_length", dsts::Series("pl", iris.numeric("petal_length")));
  check("penguins flipper_length_mm",
        dsts::Series("fl", penguins.numeric("flipper_length_mm")));
  return 0;
}