// 03_correlation.cpp
// Lesson: Pearson r measures a LINEAR relation between two numeric columns
// (-1..+1); Spearman uses ranks, so it also finds monotone relations and is
// robust to outliers. correlation_matrix() gives the whole pairwise table.
// Equivalent: df.corr(), scipy.stats.pearsonr/spearmanr.
//
// Data: iris.csv and tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## iris pearson correlation matrix
//                sepal_length  sepal_width petal_length  petal_width
//   sepal_length            1   -0.1175698    0.8717538    0.8179411
//    sepal_width   -0.1175698            1   -0.4284401   -0.3661259
//   petal_length    0.8717538   -0.4284401            1    0.9628654
//    petal_width    0.8179411   -0.3661259    0.9628654            1
//
// ## close look
// sepal_width ~ sepal_length  r = -0.1175698
// petal_width  ~ petal_length  r = 0.9628654
//
// ## tips total_bill ~ tip
// pearson  r = 0.6757341
// spearman r = 0.6789681
//
// r near 0.9 does NOT mean big bills cause big tips -- the two
// hover around similar restaurants, which is exactly the sort of
// story the correlation matrix alone cannot prove.

#include <iomanip>
#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/stats.hpp"

namespace {
void print_matrix(const dsts::DataFrame& df,
                  const std::vector<std::vector<double>>& m) {
  std::vector<std::string> cols;
  for (const auto& c : df.columns()) {
    if (df.is_numeric(c)) cols.push_back(c);
  }
  std::cout << "              ";
  for (const auto& c : cols) std::cout << std::setw(13) << c;
  std::cout << "\n";
  for (size_t i = 0; i < cols.size(); ++i) {
    std::cout << std::right << std::setw(14) << cols[i];
    for (size_t j = 0; j < cols.size(); ++j) {
      std::cout << std::setw(13) << dsts::fmt(m[i][j]);
    }
    std::cout << "\n";
  }
}
}  // namespace

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  std::cout << "## iris pearson correlation matrix\n";
  print_matrix(iris, dsts::correlation_matrix(iris));

  const dsts::Series sw = iris.series("sepal_width");
  const dsts::Series pl = iris.series("petal_length");
  std::cout << "\n## close look\n"
            << "sepal_width ~ sepal_length  r = "
            << dsts::fmt(dsts::pearson(sw, iris.series("sepal_length"))) << "\n"
            << "petal_width  ~ petal_length  r = "
            << dsts::fmt(dsts::pearson(pl, iris.series("petal_width"))) << "\n\n";

  // Spearman vs Pearson on tips.
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill = tips.series("total_bill");
  const dsts::Series tip = tips.series("tip");
  std::cout << "## tips total_bill ~ tip\n"
            << "pearson  r = " << dsts::fmt(dsts::pearson(bill, tip)) << "\n"
            << "spearman r = " << dsts::fmt(dsts::spearman(bill, tip))
            << "\n\n";
  std::cout << "r near 0.9 does NOT mean big bills cause big tips -- the two\n"
            << "hover around similar restaurants, which is exactly the sort of\n"
            << "story the correlation matrix alone cannot prove.\n";

  return 0;
}