// 04_encoding.cpp
// Lesson: ML models need numbers, not words. One-hot encoding turns one
// string column with k distinct categories into k indicator columns holding
// only 0 or 1. No category is "more important" than another.
// Equivalent: pandas.get_dummies.
//
// Data: tips.csv and iris.csv (real data).
//
// EXPECTED OUTPUT:
// ## tips smoker encoded
// 244 rows x 8 columns
// columns: total_bill tip sex smoker_No smoker_Yes day time size
//    total_bill   tip     sex  smoker_No  smoker_Yes  day    time  size
// 0       16.99  1.01  Female          1           0  Sun  Dinner     2
// 1       10.34  1.66    Male          1           0  Sun  Dinner     3
// 2       21.01   3.5    Male          1           0  Sun  Dinner     3
// 3       23.68  3.31    Male          1           0  Sun  Dinner     2
// 4       24.59  3.61  Female          1           0  Sun  Dinner     4
// 5       25.29  4.71    Male          1           0  Sun  Dinner     4
//
//
// smoker_Yes sum = 93, smoker_No sum = 151
// they always add up to the row count: 244
//
// ## iris species encoded
// 150 rows x 7 columns
// species_setosa  sum = 50 (1 flower per indicator)
// species_versicolor  sum = 50 (1 flower per indicator)
// species_virginica  sum = 50 (1 flower per indicator)

#include <iostream>

#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::DataFrame tips_enc = dsts::one_hot(tips, "smoker");
  std::cout << "## tips smoker encoded\n"
            << tips_enc.shape() << "\n";

  const auto cols = tips_enc.columns();
  std::cout << "columns:";
  for (const auto& c : cols) std::cout << " " << c;
  std::cout << "\n" << tips_enc.head(6) << "\n\n";

  {
    // total dinners per smoker value (before encoding) = per indicator sum
    const auto& yes = tips_enc.numeric("smoker_Yes");
    const auto& no = tips_enc.numeric("smoker_No");
    size_t y = 0, no_c = 0;
    for (size_t i = 0; i < yes.size(); ++i) {
      if (yes[i] && *yes[i] == 1.0) ++y;
      if (no[i] && *no[i] == 1.0) ++no_c;
    }
    std::cout << "smoker_Yes sum = " << y << ", smoker_No sum = " << no_c
              << "\n";
    std::cout << "they always add up to the row count: " << y + no_c << "\n\n";
  }

  // iris species: 3 categories -> 3 indicator columns.
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::DataFrame iris_enc = dsts::one_hot(iris, "species");
  std::cout << "## iris species encoded\n"
            << iris_enc.shape() << "\n";
  for (const auto& c : iris_enc.columns()) {
    if (c.rfind("species_", 0) != 0) continue;
    const auto& ind = iris_enc.numeric(c);
    size_t sum = 0;
    for (size_t i = 0; i < ind.size(); ++i) {
      if (ind[i] && *ind[i] == 1.0) ++sum;
    }
    std::cout << c << "  sum = " << sum << " (1 flower per indicator)\n";
  }

  return 0;
}