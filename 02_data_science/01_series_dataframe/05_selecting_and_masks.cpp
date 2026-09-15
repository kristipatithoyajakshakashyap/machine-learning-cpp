// 05_selecting_and_masks.cpp
// Lesson: a boolean mask marks which rows to keep; filter() applies one, and
// two masks combine with && (both must be true). This mimics
// df[df.col > x] and df[(a) & (b)] in pandas.
//
// Data: iris.csv and tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## iris: 150 rows, 3 species
//    sepal_length  sepal_width  petal_length  petal_width  species
// 0           5.1          3.5           1.4          0.2   setosa
// 1           4.9            3           1.4          0.2   setosa
//
//
// ## sepal_length > 6.0 keeps 61 rows
//    sepal_length     species
// 0             7  versicolor
// 1           6.4  versicolor
// 2           6.9  versicolor
// 3           6.5  versicolor
//
// ## (sepal_length > 6) & (petal_length > 4.5) keeps 53 rows
//
// ## species == "versicolor" keeps 50 rows
//    sepal_length  sepal_width  petal_length  petal_width     species
// 0             7          3.2           4.7          1.4  versicolor
// 1           6.4          3.2           4.5          1.5  versicolor
// 2           6.9          3.1           4.9          1.5  versicolor
//
//
// ## tips: total_bill>20 & tip>3 keeps 67 rows
//    total_bill   tip  day
// 0       21.01   3.5  Sun
// 1       23.68  3.31  Sun
// 2       24.59  3.61  Sun
// 3       25.29  4.71  Sun
// 4       26.88  3.12  Sun
//
// ## iris rows 0,30,60,90,120
//    sepal_length     species
// 0           5.1      setosa
// 1           4.8      setosa
// 2             5  versicolor
// 3           5.5  versicolor
// 4           6.9   virginica

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

namespace {
// Combine two element-wise masks with logical AND.
std::vector<bool> both(const std::vector<bool>& a, const std::vector<bool>& b) {
  std::vector<bool> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] && b[i];
  return out;
}
}  // namespace

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::Series sepal = iris.series("sepal_length");
  const dsts::Series petal = iris.series("petal_length");

  std::cout << "## iris: 150 rows, 3 species\n"
            << iris.head(2) << "\n\n";

  // 1. one numeric mask
  dsts::DataFrame tall = iris.filter(sepal.gt(6.0));
  std::cout << "## sepal_length > 6.0 keeps " << tall.rows() << " rows\n"
            << tall.select_columns({"sepal_length", "species"}).head(4)
            << "\n";

  // 2. two numeric masks combined
  dsts::DataFrame tall_petally =
      iris.filter(both(sepal.gt(6.0), petal.gt(4.5)));
  std::cout << "## (sepal_length > 6) & (petal_length > 4.5) keeps "
            << tall_petally.rows() << " rows\n\n";

  // 3. a string equal-to mask built from the raw string column
  std::vector<bool> versicolor(iris.rows(), false);
  const std::vector<std::string>& spec = iris.strings("species");
  for (size_t i = 0; i < spec.size(); ++i) versicolor[i] = (spec[i] == "versicolor");
  dsts::DataFrame versi = iris.filter(versicolor);
  std::cout << "## species == \"versicolor\" keeps " << versi.rows() << " rows\n"
            << versi.head(3) << "\n\n";

  // 4. chains: mask + column subset + row subset (tips)
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill = tips.series("total_bill");
  const dsts::Series tip = tips.series("tip");
  dsts::DataFrame big_meal = tips.filter(both(bill.gt(20.0), tip.gt(3.0)));
  std::cout << "## tips: total_bill>20 & tip>3 keeps " << big_meal.rows()
            << " rows\n"
            << big_meal.select_columns({"total_bill", "tip", "day"}).head(5)
            << "\n";

  // 5. every third row, by index list
  std::vector<size_t> every_third;
  for (size_t i = 0; i < iris.rows(); i += 30) every_third.push_back(i);
  std::cout << "## iris rows 0,30,60,90,120\n"
            << iris.select_rows(every_third)
                   .select_columns({"sepal_length", "species"}).head(10)
            << "\n";

  return 0;
}