// 02_stratified.cpp
// Lesson: a random split can, by luck, unbalance a rare outcome. Stratifying
// on the outcome column splits each group at the same rate, keeping the
// class mix of train and test close to the full dataset.
// Equivalent: sklearn.model_selection.StratifiedShuffleSplit.
//
// Data: titanic.csv (891 passengers; survival rate 342/891 = 0.384).
//
// EXPECTED OUTPUT:
// full      : n=891 survived 342 rate 0.3838384
//
// frac 0.7
//   naive train: n=623 survived 235 rate 0.3772071
//   naive test : n=268 survived 107 rate 0.3992537
//   strat train: n=623 survived 239 rate 0.3836276
//   strat test : n=268 survived 103 rate 0.3843284
//
// frac 0.8
//   naive train: n=712 survived 270 rate 0.3792135
//   naive test : n=179 survived 72 rate 0.4022346
//   strat train: n=712 survived 273 rate 0.383427
//   strat test : n=179 survived 69 rate 0.3854749

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"

void report(const std::string& label, const dsts::DataFrame& df) {
  size_t n = df.rows(), surv = 0;
  for (size_t r = 0; r < n; ++r) {
    if (df.cell(r, "Survived") == "1") ++surv;
  }
  std::cout << label << ": n=" << n << " survived " << surv << " rate "
            << dsts::fmt(static_cast<double>(surv) / static_cast<double>(n))
            << "\n";
}

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  report("full      ", t);

  for (double f : {0.7, 0.8}) {
    const auto [naive_train, naive_test] = dsts::train_test_split(t, f, 5);
    const auto [str_train, str_test] =
        dsts::stratified_split(t, "Survived", f, 5);
    std::cout << "\nfrac " << f << "\n";
    report("  naive train", naive_train);
    report("  naive test ", naive_test);
    report("  strat train", str_train);
    report("  strat test ", str_test);
  }
  return 0;
}