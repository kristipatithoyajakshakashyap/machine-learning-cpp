// 03_rolling_windows.cpp
// Lesson: rolling statistics slide a fixed window over an ordered series.
// With window 12 each value summarises the trailing year, which smooths
// out the seasonal wobble. Cumulative sums and ranks are the other two
// ordered-series tools you reach for constantly.
// Equivalent: s.rolling(12).mean()/.std()/.min()/.max(), s.cumsum(), s.rank().
//
// Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960).
//
// EXPECTED OUTPUT:
// rolling window 12 (first complete row is index 11)
//   month    value  mean12  std12  min12  max12  cumsum  rank
//   1949-12   118  126.6667  13.72032  104  148  1520  ...
//   ...
// wrote <results/03_rolling_windows_results>/rolling.csv
// wrote <results/03_rolling_windows_results>/rolling_mean.svg

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"
#include "dsts/datetime.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"

constexpr size_t kWindow = 12;

struct Rolling {
  std::vector<dsts::OptD> mean, sd, min, max;
};

// One pass per window: O(n * window), clear and fast enough for 144 rows.
Rolling rolling_stats(const std::vector<double>& x, size_t window) {
  Rolling r;
  const size_t n = x.size();
  r.mean.assign(n, std::nullopt);
  r.sd.assign(n, std::nullopt);
  r.min.assign(n, std::nullopt);
  r.max.assign(n, std::nullopt);
  for (size_t end = window; end <= n; ++end) {
    const size_t i = end - 1;
    double s = 0.0, lo = x[end - window], hi = lo;
    for (size_t k = end - window; k < end; ++k) {
      s += x[k];
      lo = std::min(lo, x[k]);
      hi = std::max(hi, x[k]);
    }
    const double m = s / static_cast<double>(window);
    double ss = 0.0;
    for (size_t k = end - window; k < end; ++k) ss += (x[k] - m) * (x[k] - m);
    r.mean[i] = m;
    r.sd[i] = std::sqrt(ss / static_cast<double>(window - 1));  // sample sd like pandas
    r.min[i] = lo;
    r.max[i] = hi;
  }
  return r;
}

std::vector<double> cumulative_sum(const std::vector<double>& x) {
  std::vector<double> out(x.size());
  double s = 0.0;
  for (size_t i = 0; i < x.size(); ++i) out[i] = (s += x[i]);
  return out;
}

// Average rank (1 = smallest), ties share the mean rank, like pandas rank().
std::vector<double> rank(const std::vector<double>& x) {
  std::vector<size_t> order(x.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return x[a] < x[b]; });
  std::vector<double> out(x.size());
  size_t i = 0;
  while (i < order.size()) {
    size_t j = i;
    while (j + 1 < order.size() && x[order[j + 1]] == x[order[i]]) ++j;
    const double r = 0.5 * static_cast<double>(i + j) + 1.0;
    for (size_t k = i; k <= j; ++k) out[order[k]] = r;
    i = j + 1;
  }
  return out;
}

int main() {
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const std::vector<std::string>& months = ap.strings("Month");
  std::vector<double> x;
  for (const auto& v : ap.numeric("Passengers")) x.push_back(v.value_or(0.0));
  for (const std::string& m : months)
    if (!dsts::parse_date(m)) { std::cerr << "unparsable date: " << m << "\n"; return 1; }

  const Rolling r = rolling_stats(x, kWindow);
  const std::vector<double> cs = cumulative_sum(x);
  const std::vector<double> rk = rank(x);

  std::cout << "rolling window " << kWindow << " (first complete row is index " << kWindow - 1 << ")\n"
            << "  month    value  mean12  std12  min12  max12  cumsum  rank\n";
  for (size_t i : {size_t{10}, size_t{11}, size_t{12}, size_t{23}, size_t{143}}) {
    std::cout << "  " << months[i] << "   " << dsts::fmt(x[i]) << "  " << dsts::to_text(r.mean[i])
              << "  " << dsts::to_text(r.sd[i]) << "  " << dsts::to_text(r.min[i]) << "  "
              << dsts::to_text(r.max[i]) << "  " << dsts::fmt(cs[i]) << "  " << dsts::fmt(rk[i]) << "\n";
  }
  std::cout << "  (index 10 is missing: fewer than 12 observations yet)\n"
            << "  last cumsum " << dsts::fmt(cs.back()) << " = total passengers; max rank "
            << dsts::fmt(*std::max_element(rk.begin(), rk.end())) << " = n\n";

  dsts::DataFrame out;
  out.add_string("month", months);
  out.add_numeric("value", std::vector<dsts::OptD>(x.begin(), x.end()));
  out.add_numeric("mean12", r.mean);
  out.add_numeric("std12", r.sd);
  out.add_numeric("min12", r.min);
  out.add_numeric("max12", r.max);
  out.add_numeric("cumsum", std::vector<dsts::OptD>(cs.begin(), cs.end()));
  out.add_numeric("rank", std::vector<dsts::OptD>(rk.begin(), rk.end()));
  dsts::write_csv(RUN_OUTPUT_DIR "/rolling.csv", out);
  std::cout << "wrote " RUN_OUTPUT_DIR "/rolling.csv\n";

  dsts::write_svg_line(RUN_OUTPUT_DIR "/rolling_mean.svg", "12-month rolling mean of passengers",
                       "month", "passengers", months, dsts::Series("mean12", r.mean));
  std::cout << "wrote " RUN_OUTPUT_DIR "/rolling_mean.svg\n";
  return 0;
}
