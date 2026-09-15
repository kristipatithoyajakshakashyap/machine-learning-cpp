// 02_duplicates.cpp
// Lesson: duplicated() marks every whole row that repeats an earlier one;
// drop_duplicates() keeps the first occurrence.
// Equivalent: pandas.DataFrame.duplicated() / drop_duplicates().
//
// Data: tips.csv (real data; the file contains exactly one duplicated row).
//
// EXPECTED OUTPUT:
// ## tips.csv: 244 rows x 7 columns
// duplicated rows: 1
//
// ## original (row 198):
//    total_bill  tip     sex  smoker   day   time  size
// 0          13    2  Female     Yes  Thur  Lunch     2
//
// ## duplicate (row 202):
//    total_bill  tip     sex  smoker   day   time  size
// 0          13    2  Female     Yes  Thur  Lunch     2
//
// ## drop_duplicates keeps 243 rows x 7 columns
// remaining duplicates after the fix: 0

#include <iostream>

#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  std::cout << "## tips.csv: " << tips.shape() << "\n";

  const std::vector<bool> dup = dsts::duplicated(tips);
  size_t n_dup = 0;
  for (bool d : dup) {
    if (d) ++n_dup;
  }
  std::cout << "duplicated rows: " << n_dup << "\n\n";

  // Show the duplicated row and the original it repeats.
  for (size_t i = 0; i < dup.size(); ++i) {
    if (!dup[i]) continue;
    // find the earliest row equal to row i
    size_t first = i;
    for (size_t j = 0; j < i; ++j) {
      bool same = true;
      for (const auto& c : tips.columns()) {
        if (tips.cell(i, c) != tips.cell(j, c)) {
          same = false;
          break;
        }
      }
      if (same) {
        first = j;
        break;
      }
    }
    std::cout << "## original (row " << first << "):\n"
              << tips.select_rows({first}).head(4) << "\n"
              << "## duplicate (row " << i << "):\n"
              << tips.select_rows({i}).head(4) << "\n";
  }

  const dsts::DataFrame unique = dsts::drop_duplicates(tips);
  std::cout << "## drop_duplicates keeps " << unique.shape() << "\n";

  const std::vector<bool> dup2 = dsts::duplicated(unique);
  size_t left = 0;
  for (bool d : dup2) {
    if (d) ++left;
  }
  std::cout << "remaining duplicates after the fix: " << left << "\n";

  return 0;
}