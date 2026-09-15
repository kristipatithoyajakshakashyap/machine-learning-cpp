// 03_dataframe_basics.cpp
// Lesson: a DataFrame is a labelled table of columns, like pandas.DataFrame.
// Columns are either numeric (vector<OptD>) or string; every column has the
// same row count. We build a tiny frame from the first 6 iris rows by hand.
//
// Data: rows 1-6 of iris.csv (real data).
//
// EXPECTED OUTPUT:
// ## Shape & columns
// shape   = 6 rows x 5 columns
// columns = sepal_length, sepal_width, petal_length, petal_width, species
//
// ## head(4)
//    sepal_length  sepal_width  petal_length  petal_width  species
// 0           5.1          3.5           1.4          0.2   setosa
// 1           4.9            3           1.4          0.2   setosa
// 2           4.7          3.2           1.3          0.2   setosa
// 3           4.6          3.1           1.5          0.2   setosa
//
//
// ## tail(2)
//    sepal_length  sepal_width  petal_length  petal_width  species
// 4             5          3.6           1.4          0.2   setosa
// 5           5.4          3.9           1.7          0.4   setosa
//
//
// ## info()
// DataFrame with 6 rows x 5 columns:
//   #0 sepal_length        float64   missing=0
//   #1 sepal_width         float64   missing=0
//   #2 petal_length        float64   missing=0
//   #3 petal_width         float64   missing=0
//   #4 species             str       missing=0
//
// ## cell(1, "sepal_width") = 3
// mean petal_length = 1.45, median = 1.4
//
// ## select_rows({4, 2, 0}) - is setosa at each index
//    sepal_length  species
// 0             5   setosa
// 1           4.7   setosa
// 2           5.1   setosa

#include <iostream>
#include <vector>

#include "dsts/dataframe.hpp"
#include "dsts/dtypes.hpp"

int main() {
  dsts::DataFrame iris;
  iris.add_numeric("sepal_length", {5.1, 4.9, 4.7, 4.6, 5.0, 5.4});
  iris.add_numeric("sepal_width", {3.5, 3.0, 3.2, 3.1, 3.6, 3.9});
  iris.add_numeric("petal_length", {1.4, 1.4, 1.3, 1.5, 1.4, 1.7});
  iris.add_numeric("petal_width", {0.2, 0.2, 0.2, 0.2, 0.2, 0.4});
  iris.add_string("species",
                  {"setosa", "setosa", "setosa", "setosa", "setosa", "setosa"});

  std::cout << "## Shape & columns\n"
            << "shape   = " << iris.shape() << "\n"
            << "columns = ";
  const auto cols = iris.columns();
  std::cout << cols[0];
  for (size_t i = 1; i < cols.size(); ++i) std::cout << ", " << cols[i];
  std::cout << "\n\n";

  std::cout << "## head(4)\n" << iris.head(4) << "\n\n";
  std::cout << "## tail(2)\n" << iris.tail(2) << "\n\n";
  std::cout << "## info()\n" << iris.info() << "\n";

  // one cell + one column wrapped as a Series
  std::cout << "## cell(1, \"sepal_width\") = " << iris.cell(1, "sepal_width")
            << "\n";
  dsts::Series petal = iris.series("petal_length");
  std::cout << "mean petal_length = " << dsts::fmt(petal.mean())
            << ", median = " << dsts::fmt(petal.median()) << "\n\n";

  // subsetting rows by index list, then columns by name
  dsts::DataFrame reordered_rows = iris.select_rows({4, 2, 0});
  std::cout << "## select_rows({4, 2, 0}) - is setosa at each index\n"
            << reordered_rows.select_columns({"sepal_length", "species"})
                   .head(3)
            << "\n";

  return 0;
}