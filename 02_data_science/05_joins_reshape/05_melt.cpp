// 05_melt.cpp
// Lesson: melt() is the inverse of a pivot: it stacks several numeric columns
// into "variable"/"value" pairs while repeating the id columns (long form,
// ready for plotting by groups). The iris measurements (4 numeric columns)
// become 4 x 150 = 600 rows.
// Equivalent: pandas.DataFrame.melt().
//
// Data: iris.csv (real data, 150 flowers).
//
// EXPECTED OUTPUT:
// 600 rows x 3 columns
//    species      variable  value
// 0   setosa  sepal_length    5.1
// 1   setosa  sepal_length    4.9
// 2   setosa  sepal_length    4.7
// 3   setosa  sepal_length    4.6
// 4   setosa  sepal_length      5
// 5   setosa  sepal_length    5.4
//
// sepal_length: 150 rows
// sepal_width: 150 rows
// petal_length: 150 rows
// petal_width: 150 rows
// overall mean of all 600 values: 3.4645

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/join.hpp"

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");

  const dsts::DataFrame long_form = dsts::melt(
      iris, {"species"},
      {"sepal_length", "sepal_width", "petal_length", "petal_width"});

  std::cout << long_form.shape() << "\n";
  std::cout << long_form.head(6) << "\n";

  // rows per variable (each measurement should appear 150 times)
  const std::vector<std::string> vars = {"sepal_length", "sepal_width",
                                         "petal_length", "petal_width"};
  for (const std::string& v : vars) {
    int n = 0;
    for (size_t r = 0; r < long_form.rows(); ++r) {
      if (long_form.strings("variable")[r] == v) ++n;
    }
    std::cout << v << ": " << n << " rows\n";
  }

  // mean petal_length value in long form vs the original column
  double sum = 0.0;
  int n = 0;
  for (size_t r = 0; r < long_form.rows(); ++r) {
    sum += *long_form.numeric("value")[r];
    ++n;
  }
  std::cout << "overall mean of all 600 values: "
            << dsts::fmt(sum / n) << "\n";
  return 0;
}