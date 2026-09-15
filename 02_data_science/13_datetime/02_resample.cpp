// 02_resample.cpp
// Lesson: resampling is "group by a coarser calendar key". Monthly passenger
// counts become quarterly and yearly totals/means by grouping on
// quarter_key ("1949-Q1") and the year, then summing or averaging.
// Equivalent: s.resample("QS").sum(), s.resample("YS").mean().
//
// Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960).
//
// EXPECTED OUTPUT:
// quarterly (first 8 of 48): key  n  sum  mean
//   1949-Q1  3  362  120.6667
//   1949-Q2  3  385  128.3333
//   ...
// yearly (12): key  n  sum  mean
//   1949  12  1520  126.6667
//   ...
//   1960  12  5714  476.1667
// wrote <results/02_resample_results>/resampled.csv

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"
#include "dsts/datetime.hpp"

struct Bucket {
  size_t n = 0;
  double sum = 0.0;
  double mean() const { return n ? sum / static_cast<double>(n) : 0.0; }
};

// Group values by the key produced from each date. std::map keeps keys sorted,
// and "YYYY-Qn" / "YYYY" sort chronologically as text.
using Buckets = std::map<std::string, Bucket>;

template <typename KeyFn>
Buckets resample(const std::vector<dsts::Date>& dates, const std::vector<dsts::OptD>& values,
                 KeyFn key_of) {
  Buckets out;
  for (size_t i = 0; i < dates.size(); ++i) {
    if (!values[i].has_value()) continue;
    Bucket& b = out[key_of(dates[i])];
    ++b.n;
    b.sum += *values[i];
  }
  return out;
}

void print(const std::string& title, const Buckets& b, size_t limit) {
  std::cout << title << ": key  n  sum  mean\n";
  size_t shown = 0;
  for (const auto& [key, bucket] : b) {
    if (shown++ == limit) { std::cout << "  ...\n"; break; }
    std::cout << "  " << key << "  " << bucket.n << "  " << dsts::fmt(bucket.sum) << "  "
              << dsts::fmt(bucket.mean()) << "\n";
  }
}

void append(const std::string& freq, const Buckets& b,
            std::vector<std::string>& freqs, std::vector<std::string>& keys,
            std::vector<dsts::OptD>& ns, std::vector<dsts::OptD>& sums,
            std::vector<dsts::OptD>& means) {
  for (const auto& [key, bucket] : b) {
    freqs.push_back(freq);
    keys.push_back(key);
    ns.push_back(static_cast<double>(bucket.n));
    sums.push_back(bucket.sum);
    means.push_back(bucket.mean());
  }
}

int main() {
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const std::vector<dsts::OptD>& passengers = ap.numeric("Passengers");

  std::vector<dsts::Date> dates;
  for (const std::string& t : ap.strings("Month")) {
    const auto d = dsts::parse_date(t);
    if (!d) { std::cerr << "unparsable date: " << t << "\n"; return 1; }
    dates.push_back(*d);
  }

  const Buckets quarterly = resample(dates, passengers, dsts::quarter_key);
  const Buckets yearly = resample(dates, passengers,
                                  [](dsts::Date d) { return std::to_string(d.year); });
  print("quarterly (first 8 of " + std::to_string(quarterly.size()) + ")", quarterly, 8);
  print("yearly (" + std::to_string(yearly.size()) + ")", yearly, 12);

  // Sanity: every bucket is complete (3 months per quarter, 12 per year) and
  // the totals agree with the raw sum.
  double raw_total = 0.0;
  for (const auto& v : passengers) raw_total += v.value_or(0.0);
  double q_total = 0.0, y_total = 0.0;
  for (const auto& [k, b] : quarterly) q_total += b.sum;
  for (const auto& [k, b] : yearly) y_total += b.sum;
  std::cout << "\ntotals: raw " << dsts::fmt(raw_total) << "  quarterly " << dsts::fmt(q_total)
            << "  yearly " << dsts::fmt(y_total) << (raw_total == q_total && raw_total == y_total ? "  (match)" : "  (MISMATCH)")
            << "\n";

  dsts::DataFrame out;
  std::vector<std::string> freqs, keys;
  std::vector<dsts::OptD> ns, sums, means;
  append("Q", quarterly, freqs, keys, ns, sums, means);
  append("Y", yearly, freqs, keys, ns, sums, means);
  out.add_string("freq", freqs);
  out.add_string("period", keys);
  out.add_numeric("n", ns);
  out.add_numeric("sum", sums);
  out.add_numeric("mean", means);
  dsts::write_csv(RUN_OUTPUT_DIR "/resampled.csv", out);
  std::cout << "wrote " RUN_OUTPUT_DIR "/resampled.csv\n";
  return 0;
}
