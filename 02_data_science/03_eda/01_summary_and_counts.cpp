// 01_summary_and_counts.cpp
// Lesson: EDA starts with summaries. summary() prints pandas-style
// count/mean/std/min/quantiles/max for every numeric column; value_counts()
// tallies a string column.
// Equivalent: df.describe(), df.value_counts().
//
// Data: iris.csv and tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## iris numeric summary
//             sepal_length  sepal_width  petal_length  petal_width
// count                150          150           150          150
// mean            5.843333     3.057333         3.758     1.199333
// std            0.8280661    0.4358663      1.765298    0.7622377
// min                  4.3            2             1          0.1
// 25%                  5.1          2.8           1.6          0.3
// 50%                  5.8            3          4.35          1.3
// 75%                  6.4          3.3           5.1          1.8
// max                  7.9          4.4           6.9          2.5
//
// ## tips smoker counts
//   No  151
//   Yes  93
//
// ## tips day counts
//   Sat  87
//   Sun  76
//   Thur  62
//   Fri  19
//
// ## iris species counts
//   setosa  50
//   versicolor  50
//   virginica  50
//
// all counts are integers; sizes = 150 iris rows

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  std::cout << "## iris numeric summary\n" << dsts::summary(iris) << "\n";

  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  std::cout << "## tips smoker counts\n";
  for (const auto& kv : dsts::value_counts(tips, "smoker")) {
    std::cout << "  " << kv.first << "  " << kv.second << "\n";
  }
  std::cout << "\n## tips day counts\n";
  for (const auto& kv : dsts::value_counts(tips, "day")) {
    std::cout << "  " << kv.first << "  " << kv.second << "\n";
  }
  std::cout << "\n## iris species counts\n";
  for (const auto& kv : dsts::value_counts(iris, "species")) {
    std::cout << "  " << kv.first << "  " << kv.second << "\n";
  }
  std::cout << "\nall counts are integers; sizes = " << iris.rows() << " iris rows\n";

  return 0;
}