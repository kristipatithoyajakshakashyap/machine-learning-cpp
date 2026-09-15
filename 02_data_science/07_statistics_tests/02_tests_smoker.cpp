// 02_tests_smoker.cpp
// Lesson: Welch's t-test compares two group means under unequal variances.
// Both the tip and the total bill are compared between smokers and
// non-smokers; small p-values indicate the difference is not just noise.
// Equivalent: scipy.stats.ttest_ind(equal_var=False).
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// ## total_bill: smoker No vs Yes
//   No : mean 19.18828  sd 8.255582  n 151
//   Yes: mean 20.75634  sd 9.832154  n 93
//   t=-1.284252  df=169.6259  p=0.2008052
//
// ## tip: smoker No vs Yes
//   No : mean 2.991854  sd 1.37719  n 151
//   Yes: mean 3.00871  sd 1.401468  n 93
//   t=-0.0918442  df=192.2634  p=0.9269174

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

dsts::Series by_smoker(const dsts::DataFrame& tips, const std::string& group,
                       const std::string& col) {
  std::vector<dsts::OptD> vals;
  for (size_t r = 0; r < tips.rows(); ++r) {
    if (tips.strings("smoker")[r] == group) vals.push_back(tips.numeric(col)[r]);
  }
  return dsts::Series(group + "_" + col, vals);
}

void compare(const dsts::DataFrame& tips, const std::string& col) {
  const dsts::Series no = by_smoker(tips, "No", col);
  const dsts::Series yes = by_smoker(tips, "Yes", col);
  const dsts::TTestResult r = dsts::ttest_ind(no, yes);
  std::cout << "## " << col << ": smoker No vs Yes\n";
  std::cout << "  No : mean " << dsts::fmt(no.mean()) << "  sd "
            << dsts::fmt(no.stddev()) << "  n " << no.count() << "\n";
  std::cout << "  Yes: mean " << dsts::fmt(yes.mean()) << "  sd "
            << dsts::fmt(yes.stddev()) << "  n " << yes.count() << "\n";
  std::cout << "  t=" << dsts::fmt(r.t) << "  df=" << dsts::fmt(r.df)
            << "  p=" << dsts::fmt(r.p_value) << "\n\n";
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  compare(tips, "total_bill");
  compare(tips, "tip");
  return 0;
}