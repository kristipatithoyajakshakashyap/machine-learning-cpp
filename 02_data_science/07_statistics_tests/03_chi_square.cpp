// 03_chi_square.cpp
// Lesson: the chi-square independence test checks whether two categorical
// columns are associated. Survival x Sex and survival x passenger class are
// tested on the real titanic file. chi2_independence() takes the observed
// contingency table (rows x columns) and returns statistic, degrees of
// freedom and p-value.
// Equivalent: scipy.stats.chi2_contingency.
//
// Data: titanic.csv (real data, 891 passengers).
//
// EXPECTED OUTPUT:
// ## Sex x Survived
//   male: survived 109 / total 577
//   female: survived 233 / total 314
//   chi2=263.0506  dof=1  p=3.711748e-59
//
// ## Pclass x Survived
//   1: survived 136 / total 216
//   2: survived 87 / total 184
//   3: survived 119 / total 491
//   chi2=102.889  dof=2  p=4.549252e-23

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/stats.hpp"

typedef dsts::ChiSquareResult CSR;

void contingency(const dsts::DataFrame& t, const std::string& row_col,
                 const std::vector<std::string>& rows,
                 const std::vector<std::string>& cols) {
  std::vector<std::vector<double>> observed(
      rows.size(), std::vector<double>(cols.size(), 0.0));
  for (size_t r = 0; r < t.rows(); ++r) {
    size_t ri = 0, ci = 0;
    for (size_t i = 0; i < rows.size(); ++i)
      if (rows[i] == t.cell(r, row_col)) ri = i;
    for (size_t i = 0; i < cols.size(); ++i)
      if (cols[i] == t.cell(r, "Survived")) ci = i;
    observed[ri][ci] += 1.0;
  }
  const CSR res = dsts::chi2_independence(observed);
  std::cout << "## " << row_col << " x Survived\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    std::cout << "  " << rows[i] << ": survived " << int(observed[i][1])
              << " / total " << int(observed[i][0] + observed[i][1]) << "\n";
  }
  std::cout << "  chi2=" << dsts::fmt(res.chi2) << "  dof=" << res.dof
            << "  p=" << dsts::fmt(res.p_value) << "\n\n";
}

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  contingency(t, "Sex", {"male", "female"}, {"0", "1"});
  contingency(t, "Pclass", {"1", "2", "3"}, {"0", "1"});
  return 0;
}