// 02_data_science/src/eda.cpp
// dsts exploratory-data-analysis helpers: value counts, a pandas-style
// describe() table, histograms, a Pearson correlation matrix and grouped
// mean/sd statistics.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 03_eda, 11_final_pipeline and 12_advanced_eda.
// Exercised by ctest -R dsts_regression.

#include "dsts/eda.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

#include "dsts/stats.hpp"

namespace dsts {

// Count each distinct string in `col`. Sorted by count descending, then
// label ascending so the output is deterministic.
std::vector<std::pair<std::string, size_t>> value_counts(
    const DataFrame& df, const std::string& col) {
  const auto& all = df.strings(col);
  std::map<std::string, size_t> tally;
  for (const auto& v : all) ++tally[v];
  std::vector<std::pair<std::string, size_t>> out(tally.begin(), tally.end());
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) return a.second > b.second;
              return a.first < b.first;
            });
  return out;
}

// describe()-style table for every numeric column: count, mean, std, min,
// quartiles, max. Count is the number of present values; std is the Series
// sample std-dev. Returns the formatted text.
std::string summary(const DataFrame& df) {
  const auto cols = df.columns();
  std::vector<std::string> numeric_cols;
  for (const auto& c : cols) {
    if (df.is_numeric(c)) numeric_cols.push_back(c);
  }
  size_t w = 12;  // widest stat label, e.g. "         25%"
  for (const auto& c : numeric_cols) w = std::max(w, c.size());
  std::ostringstream os;
  os << std::left << std::setw(static_cast<int>(w)) << "";
  for (const auto& c : numeric_cols) {
    os << std::right << std::setw(static_cast<int>(c.size())) << c << "  ";
  }
  os << "\n";
  const std::vector<std::string> rows = {"count", "mean", "std",
                                         "min",   "25%",  "50%",
                                         "75%",   "max"};
  for (const auto& label : rows) {
    os << std::left << std::setw(static_cast<int>(w)) << label;
    for (const auto& c : numeric_cols) {
      const Series s = df.series(c);
      double v = 0.0;
      if (label == "count") v = static_cast<double>(s.size());
      if (label == "mean") v = s.mean();
      if (label == "std") v = s.stddev();
      if (label == "min") v = s.min();
      if (label == "25%") v = s.quantile(0.25);
      if (label == "50%") v = s.quantile(0.50);
      if (label == "75%") v = s.quantile(0.75);
      if (label == "max") v = s.max();
      os << std::right << std::setw(static_cast<int>(c.size())) << fmt(v)
         << "  ";
    }
    os << "\n";
  }
  return os.str();
}

// Equal-width histogram over [min, max] with `bins` bins. Missing values
// are skipped; the maximum lands in the last bin so the top edge is inclusive.
// Empty series or bins == 0 gives an empty Histogram.
Histogram histogram(const Series& x, size_t bins) {
  Histogram h;
  if (x.size() == 0 || bins == 0) return h;
  const double lo = x.min();
  const double hi = x.max();
  h.edges.reserve(bins + 1);
  h.counts.assign(bins, 0);
  const double span = hi - lo;
  for (size_t b = 0; b <= bins; ++b) {
    h.edges.push_back(lo + span * static_cast<double>(b) / static_cast<double>(bins));
  }
  const auto values = x.values();
  for (const OptD& v : values) {
    if (!v.has_value()) continue;
    const double t = *v;
    if (t == hi) {  // top edge belongs in the last bin
      ++h.counts[bins - 1];
      continue;
    }
    size_t b = 0;
    if (span > 0.0) {
      // Bin index by proportion of the span; clamp guards rounding at the edge.
      b = static_cast<size_t>((t - lo) / span * static_cast<double>(bins));
      if (b >= bins) b = bins - 1;
    }
    ++h.counts[b];
  }
  return h;
}

// Pairwise Pearson correlation of every numeric column, in column order.
// The result is symmetric with 1.0 on the diagonal (for non-constant columns).
std::vector<std::vector<double>> correlation_matrix(const DataFrame& df) {
  std::vector<std::string> cols;
  for (const auto& c : df.columns()) {
    if (df.is_numeric(c)) cols.push_back(c);
  }
  std::vector<std::vector<double>> m(cols.size(),
                                     std::vector<double>(cols.size(), 0.0));
  for (size_t i = 0; i < cols.size(); ++i) {
    for (size_t j = 0; j < cols.size(); ++j) {
      m[i][j] = pearson(df.series(cols[i]), df.series(cols[j]));
    }
  }
  return m;
}

// Mean and sample sd of `value_col` per distinct value of string `group_col`.
// Missing values are excluded; groups are returned in sorted label order;
// sd is NaN for a group with a single observation.
std::vector<GroupStat> group_mean(const DataFrame& df,
                                  const std::string& group_col,
                                  const std::string& value_col) {
  const auto& groups = df.strings(group_col);
  const auto& values = df.numeric(value_col);
  std::map<std::string, std::vector<double>> buckets;
  for (size_t i = 0; i < groups.size(); ++i) {
    if (values[i].has_value()) buckets[groups[i]].push_back(*values[i]);
  }
  std::vector<GroupStat> out;
  for (const auto& kv : buckets) {
    GroupStat g;
    g.group = kv.first;
    g.count = kv.second.size();
    double sum = 0.0;
    for (double v : kv.second) sum += v;
    g.mean = sum / static_cast<double>(kv.second.size());
    double ss = 0.0;
    for (double v : kv.second) ss += (v - g.mean) * (v - g.mean);
    g.sd = kv.second.size() > 1 ? std::sqrt(ss / static_cast<double>(kv.second.size() - 1))
                                : std::numeric_limits<double>::quiet_NaN();
    out.push_back(g);
  }
  return out;
}

}  // namespace dsts