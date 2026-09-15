// 02_inner_vs_left.cpp
// Lesson: inner joins keep only matching rows; left joins keep every left row
// and fill unmatched cells with NA. Both are shown on real passenger-data
// slices of the titanic file (passengerId is unique, so keys never repeat).
// Equivalent: pandas.merge(how="inner") vs how="left".
//
// Data: titanic.csv (real data, 891 passengers).
//
// EXPECTED OUTPUT:
// ## inner join keeps only matches
// 10 rows x 5 columns
//    PassengerId              Name  Survived  Age  Cabin
// 0            1  Braund, Mr. Owe~         0   22
// 1            2  Cumings, Mrs. J~         1   38    C85
// 2            3  Heikkinen, Miss~         1   26
// 3            4  Futrelle, Mrs. ~         1   35   C123
//    .
// 6            7  McCarthy, Mr. T~         0   54    E46
// 7            8  Palsson, Master~         0    2
// 8            9  Johnson, Mrs. O~         1   27
// 9           10  Nasser, Mrs. Ni~         1   14
//
// ## left join keeps all 12 left rows (NA on the last two)
// 12 rows x 5 columns
//     PassengerId              Name  Survived  Age  Cabin
//  0            1  Braund, Mr. Owe~         0   22
//  1            2  Cumings, Mrs. J~         1   38    C85
//  2            3  Heikkinen, Miss~         1   26
//  3            4  Futrelle, Mrs. ~         1   35   C123
//     .
//  8            9  Johnson, Mrs. O~         1   27
//  9           10  Nasser, Mrs. Ni~         1   14
// 10           11  Sandstrom, Miss~         1   NA
// 11           12  Bonnell, Miss. ~         1   NA

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"
#include "dsts/join.hpp"

int main() {
  const dsts::DataFrame titanic = dsts::read_csv(DATA_DIR "/titanic.csv");

  // Left frame: first 12 passengers, just id/name/survived.
  std::vector<size_t> left_idx;
  for (size_t i = 0; i < 12; ++i) left_idx.push_back(i);
  const dsts::DataFrame left =
      titanic.select_rows(left_idx)
          .select_columns({"PassengerId", "Name", "Survived"});

  // Right frame: first 10 passengers with age + cabin.
  std::vector<size_t> right_idx;
  for (size_t i = 0; i < 10; ++i) right_idx.push_back(i);
  const dsts::DataFrame right =
      titanic.select_rows(right_idx)
          .select_columns({"PassengerId", "Age", "Cabin"});

  const dsts::DataFrame inner =
      dsts::merge(left, right, "PassengerId", "inner");
  const dsts::DataFrame left_join =
      dsts::merge(left, right, "PassengerId", "left");

  std::cout << "## inner join keeps only matches\n"
            << inner.shape() << "\n"
            << inner.to_string(8) << "\n";

  std::cout << "## left join keeps all 12 left rows (NA on the last two)\n"
            << left_join.shape() << "\n"
            << left_join.to_string(8) << "\n";
  return 0;
}