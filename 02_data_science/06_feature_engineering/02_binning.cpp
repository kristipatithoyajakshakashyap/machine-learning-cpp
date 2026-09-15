// 02_binning.cpp
// Lesson: cut() turns a continuous column into ordered categories with
// explicit edges; qcut_edges() finds edges that give equal-count bins.
// Two real examples: passenger Age into child/teen/adult/senior, and the
// tip column into equal-count quartiles.
// Equivalent: pandas.cut, pandas.qcut.
//
// Data: titanic.csv (real data) and tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## titanic Age groups (891 passengers)
//   adult: 502
//   NA: 177
//   teen: 133
//   child: 71
//   senior: 8
// ## total_bill quartile edges
//   3.07  13.3475  17.795  24.1275  50.81
//   Q1: 61
//   Q2: 61
//   Q3: 61
//   Q4: 61

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/features.hpp"
#include "dsts/series.hpp"

int main() {
  const dsts::DataFrame titanic = dsts::read_csv(DATA_DIR "/titanic.csv");
  const dsts::Series age("Age", titanic.numeric("Age"));

  const auto groups = dsts::cut(age, {0, 13, 21, 65, 100},
                                {"child", "teen", "adult", "senior"});
  dsts::DataFrame age_df;
  age_df.add_string("age_group", groups);
  std::cout << "## titanic Age groups (891 passengers)\n";
  for (const auto& kv : dsts::value_counts(age_df, "age_group")) {
    std::cout << "  " << kv.first << ": " << kv.second << "\n";
  }

  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series total_bill("total_bill", tips.numeric("total_bill"));
  const auto edges = dsts::qcut_edges(total_bill, 4);
  std::cout << "## total_bill quartile edges\n";
  for (double e : edges) std::cout << "  " << dsts::fmt(e);
  std::cout << "\n";

  const auto qbins = dsts::cut(total_bill, edges, {"Q1", "Q2", "Q3", "Q4"});
  dsts::DataFrame bill_df;
  bill_df.add_string("quartile", qbins);
  for (const auto& kv : dsts::value_counts(bill_df, "quartile")) {
    std::cout << "  " << kv.first << ": " << kv.second << "\n";
  }
  return 0;
}