// 01_confidence_intervals.cpp
// Lesson: a 95% confidence interval for a sample mean shows the range the
// true population mean plausibly lives in. mean_ci95 uses the normal
// approximation (z = 1.96) - fine at n = 244 or 87.
// Equivalent: scipy.stats.t.interval / seaborn's error bars.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// total_bill  n=244  mean=19.78594  std=8.902412  95%CI [18.66892, 20.90296]  width=2.23404
// tip  n=244  mean=2.998279  std=1.383638  95%CI [2.824668, 3.171889]  width=0.3472208
// tip Thur  n=62  mean=2.771452  std=1.240223  95%CI [2.462741, 3.080163]  width=0.617422
// tip Fri  n=19  mean=2.734737  std=1.019577  95%CI [2.276287, 3.193186]  width=0.9168987
// tip Sat  n=87  mean=2.993103  std=1.631014  95%CI [2.650378, 3.335829]  width=0.6854507
// tip Sun  n=76  mean=3.255132  std=1.23488  95%CI [2.977502, 3.532761]  width=0.5552597

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

void report(const dsts::Series& x) {
  const dsts::ConfidenceInterval ci = dsts::mean_ci95(x);
  std::cout << x.name() << "  n=" << x.count() << "  mean="
            << dsts::fmt(x.mean()) << "  std=" << dsts::fmt(x.stddev())
            << "  95%CI [" << dsts::fmt(ci.lower) << ", " << dsts::fmt(ci.upper)
            << "]  width=" << dsts::fmt(ci.upper - ci.lower) << "\n";
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  report(dsts::Series("total_bill", tips.numeric("total_bill")));
  report(dsts::Series("tip", tips.numeric("tip")));

  // Day-level means: build one Series per day and give each a CI.
  for (const char* day_s : {"Thur", "Fri", "Sat", "Sun"}) {
    const std::string day(day_s);
    std::vector<dsts::OptD> vals;
    for (size_t r = 0; r < tips.rows(); ++r) {
      if (tips.strings("day")[r] == day) vals.push_back(tips.numeric("tip")[r]);
    }
    report(dsts::Series("tip " + day, vals));
  }
  return 0;
}