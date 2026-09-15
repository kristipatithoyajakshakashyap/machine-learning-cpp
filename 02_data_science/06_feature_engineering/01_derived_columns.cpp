// 01_derived_columns.cpp
// Lesson: raw columns are rarely the best predictors. New features are
// computed from them with elementwise Series arithmetic: bill per person and
// tip as a percentage of the bill for the real tips file.
// Equivalent: df["bill_per_person"] = df.total_bill / df.size.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// ## bill_per_person
// mean 7.88823  std 2.91435  min 2.875  max 20.275
// ## tip_pct (%)
// mean 16.08026  std 6.10722
// ## engineered head
//    day  bill_per_person   tip_pct
// 0  Sun            8.495  5.944673
// 1  Sun         3.446667  16.05416
// 2  Sun         7.003333  16.65873
// 3  Sun            11.84  13.97804
// 4  Sun           6.1475  14.68076
//
// Fri  mean tip_pct 16.9913  (n 19)
// Sat  mean tip_pct 15.31517  (n 87)
// Sun  mean tip_pct 16.68973  (n 76)
// Thur  mean tip_pct 16.12756  (n 62)

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/series.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill("total_bill", tips.numeric("total_bill"));
  const dsts::Series tip("tip", tips.numeric("tip"));
  const dsts::Series size_s("size", tips.numeric("size"));

  const dsts::Series bill_per_person = bill.div(size_s);
  const dsts::Series tip_pct = tip.div(bill).mul(100.0);

  std::cout << "## bill_per_person\nmean " << dsts::fmt(bill_per_person.mean())
            << "  std " << dsts::fmt(bill_per_person.stddev())
            << "  min " << dsts::fmt(bill_per_person.min())
            << "  max " << dsts::fmt(bill_per_person.max()) << "\n";
  std::cout << "## tip_pct (%)\nmean " << dsts::fmt(tip_pct.mean())
            << "  std " << dsts::fmt(tip_pct.stddev()) << "\n";

  dsts::DataFrame eng;
  std::vector<std::string> days;
  for (size_t r = 0; r < tips.rows(); ++r) days.push_back(tips.strings("day")[r]);
  eng.add_string("day", days);
  eng.add_numeric("bill_per_person", bill_per_person.values());
  eng.add_numeric("tip_pct", tip_pct.values());
  std::cout << "## engineered head\n" << eng.head(5) << "\n";

  for (const auto& g : dsts::group_mean(eng, "day", "tip_pct")) {
    std::cout << g.group << "  mean tip_pct " << dsts::fmt(g.mean) << "  (n "
              << g.count << ")\n";
  }
  return 0;
}