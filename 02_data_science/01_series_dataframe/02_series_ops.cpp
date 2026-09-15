// 02_series_ops.cpp
// Lesson: arithmetic on a Series maps over the cells, and a missing cell
// stays missing. Comparison masks feed straight into DataFrame::filter
// (lesson 05).
//
// Data: first 10 penguin measurements from penguins.csv (real data).
// Row 4 of the file is one of the two "all four measurements missing"
// Adelie penguins, so bill_length has a genuine missing cell here.
//
// EXPECTED OUTPUT:
// ## Raw bill lengths (mm)
// 0    39.1
// 1    39.5
// 2    40.3
// 3    NA
// 4    36.7
// 5    39.3
// 6    38.9
// 7    39.2
// 8    34.1
// 9    42
// Name: bill_length_mm, dtype: float64
// count   = 9
// missing = 1
//
// ## bill_length_mm / 10 (cm)
// 0    3.91
// 1    3.95
// 2    4.03
// 3    NA
// 4    3.67
// Name: bill_length_mm, dtype: float64
//
// ## body_mass_g / 1000 (kg) - missing stays missing
// 0    3.75
// 1    3.8
// 2    3.25
// 3    NA
// 4    3.45
// 5    3.65
// Name: body_mass_g, dtype: float64
//
// mean(bill_length)          = 38.78889
// mean(bill_length / 10)     = 3.878889
//
// ## Boolean mask bill_length > 38
// 0  true   39.1
// 1  true   39.5
// 2  true   40.3
// 3  false   NA
// 4  false   36.7
// 5  true   39.3
// 6  true   38.9
// 7  true   39.2
// 8  false   34.1
// 9  true   42
// penguins with bill > 38 mm: 7

#include <iostream>
#include <vector>

#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"

int main() {
  // bill_length_mm for the first 10 penguins; row 4 (index 3) is missing.
  std::vector<dsts::OptD> bill({39.1, 39.5, 40.3, std::nullopt,
                                36.7, 39.3, 38.9, 39.2, 34.1, 42.0});
  dsts::Series bill_length("bill_length_mm", bill);

  std::cout << "## Raw bill lengths (mm)\n" << bill_length.to_string(10) << "\n";
  std::cout << "count   = " << bill_length.count() << "\n"
            << "missing = " << bill_length.missing() << "\n\n";

  // 1. scalar arithmetic: millimetres -> centimetres
  dsts::Series in_cm = bill_length.div(10.0);
  std::cout << "## bill_length_mm / 10 (cm)\n" << in_cm.head(5) << "\n\n";

  // 2. NaN propagation: body mass in kg from first 6 penguins, row 4 missing
  std::vector<dsts::OptD> mass({3750.0, 3800.0, 3250.0, std::nullopt,
                                3450.0, 3650.0});
  dsts::Series body_mass("body_mass_g", mass);
  dsts::Series in_kg = body_mass.div(1000.0);
  std::cout << "## body_mass_g / 1000 (kg) - missing stays missing\n"
            << in_kg.to_string(10) << "\n\n";

  // 3. the mean of the transformed series
  std::cout << "mean(bill_length)          = "
            << dsts::fmt(bill_length.mean()) << "\n"
            << "mean(bill_length / 10)     = "
            << dsts::fmt(in_cm.mean()) << "\n\n";

  // 4. comparison masks: how many of the 10 penguins exceed 38 mm?
  const std::vector<bool> big = bill_length.gt(38.0);
  int big_count = 0;
  for (size_t i = 0; i < big.size(); ++i) {
    if (big[i]) ++big_count;
  }
  std::cout << "## Boolean mask bill_length > 38\n";
  for (size_t i = 0; i < big.size(); ++i) {
    std::cout << i << "  " << (big[i] ? "true" : "false")
              << "   " << dsts::to_text(bill.at(i)) << "\n";
  }
  std::cout << "penguins with bill > 38 mm: " << big_count << "\n";

  return 0;
}