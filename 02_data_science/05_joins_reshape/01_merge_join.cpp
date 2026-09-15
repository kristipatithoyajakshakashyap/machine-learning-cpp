// 01_merge_join.cpp
// Lesson: merge() attaches columns from a lookup table using a shared key.
// Here a day-level table (average bill + average tip per day, aggregated
// from the same real file) is joined onto every individual tip by "day".
// Equivalent: pandas.merge(tips, day_stats, on="day", how="left").
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// ## day lookup table
//     day  avg_bill   avg_tip
// 0   Fri  17.15158  2.734737
// 1   Sat  20.44138  2.993103
// 2   Sun     21.41  3.255132
// 3  Thur  17.68274  2.771452
//
// ## tips joined with day averages
// 244 rows x 9 columns
//    total_bill   tip     sex  smoker  day    time  size  avg_bill   avg_tip
// 0       16.99  1.01  Female      No  Sun  Dinner     2     21.41  3.255132
// 1       10.34  1.66    Male      No  Sun  Dinner     3     21.41  3.255132
// 2       21.01   3.5    Male      No  Sun  Dinner     3     21.41  3.255132
// 3       23.68  3.31    Male      No  Sun  Dinner     2     21.41  3.255132
//
// first row avg_bill = 21.41  avg_tip = 3.255132

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/join.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");

  // Build the day-level lookup table from the same real data.
  std::map<std::string, std::pair<double, double>> acc;  // total_bill, tip
  std::map<std::string, int> cnt;
  for (size_t r = 0; r < tips.rows(); ++r) {
    const std::string day = tips.strings("day")[r];
    acc[day].first += *tips.numeric("total_bill")[r];
    acc[day].second += *tips.numeric("tip")[r];
    cnt[day] += 1;
  }
  std::vector<std::string> days;
  std::vector<dsts::OptD> avg_bill, avg_tip;
  for (const auto& kv : acc) {
    days.push_back(kv.first);
    avg_bill.push_back(kv.second.first / cnt[kv.first]);
    avg_tip.push_back(kv.second.second / cnt[kv.first]);
  }
  dsts::DataFrame day_stats;
  day_stats.add_string("day", days);
  day_stats.add_numeric("avg_bill", avg_bill);
  day_stats.add_numeric("avg_tip", avg_tip);

  std::cout << "## day lookup table\n" << day_stats.to_string() << "\n";

  const dsts::DataFrame merged =
      dsts::merge(tips, day_stats, "day", "left");
  std::cout << "## tips joined with day averages\n"
            << merged.shape() << "\n"
            << merged.head(4) << "\n";

  std::cout << "first row avg_bill = " << dsts::fmt(*merged.numeric("avg_bill")[0])
            << "  avg_tip = " << dsts::fmt(*merged.numeric("avg_tip")[0])
            << "\n";
  return 0;
}