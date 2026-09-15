// 02_data_science/src/timeseries.cpp
// dsts time-series helpers: trailing rolling mean, lag autocorrelation,
// first difference and RMSE between two aligned series.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 09_time_series. Exercised by ctest -R dsts_regression.

#include "dsts/timeseries.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace dsts {

// Trailing moving average of width `window`. The first window-1 outputs are
// missing. Uses a running sum so the loop is O(n). Assumes every input value
// is present (a missing cell would dereference an empty optional).
Series rolling_mean(const Series& x, size_t window) {
  if (window == 0) throw std::invalid_argument("rolling_mean window must be > 0");
  const auto vals = x.values();
  std::vector<OptD> out(vals.size(), std::nullopt);
  if (vals.size() >= window) {
    double sum = 0.0;
    for (size_t i = 0; i + 1 < window; ++i) sum += *vals[i];
    for (size_t i = window - 1; i < vals.size(); ++i) {
      sum += *vals[i];
      out[i] = sum / static_cast<double>(window);
      // Drop the element that leaves the window before the next iteration.
      sum -= *vals[i + 1 - window];
    }
  }
  return Series(x.name() + "_ma" + std::to_string(window), out);
}

// Pearson correlation between x[t] and x[t-lag] over pairs where both are
// present. NaN when lag is 0, too large, fewer than 2 pairs exist, or either
// side is constant.
double autocorr(const Series& x, size_t lag) {
  const auto vals = x.values();
  if (lag == 0 || vals.size() <= lag) return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> head, tail;
  for (size_t i = lag; i < vals.size(); ++i) {
    if (vals[i].has_value() && vals[i - lag].has_value()) {
      head.push_back(*vals[i - lag]);
      tail.push_back(*vals[i]);
    }
  }
  if (head.size() < 2) return std::numeric_limits<double>::quiet_NaN();
  double mx = 0, my = 0;
  for (size_t i = 0; i < head.size(); ++i) {
    mx += head[i];
    my += tail[i];
  }
  mx /= static_cast<double>(head.size());
  my /= static_cast<double>(head.size());
  double num = 0, sx = 0, sy = 0;
  for (size_t i = 0; i < head.size(); ++i) {
    const double dx = head[i] - mx, dy = tail[i] - my;
    num += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  if (sx == 0.0 || sy == 0.0) return std::numeric_limits<double>::quiet_NaN();
  return num / std::sqrt(sx * sy);
}

// x[t] - x[t-1]; the first element and any pair with a missing side is missing.
Series diff(const Series& x) {
  const auto vals = x.values();
  std::vector<OptD> out(vals.size(), std::nullopt);
  for (size_t i = 1; i < vals.size(); ++i) {
    if (vals[i].has_value() && vals[i - 1].has_value()) {
      out[i] = *vals[i] - *vals[i - 1];
    }
  }
  return Series(x.name() + "_diff", out);
}

// Root-mean-square error over positions where both values are present.
// Lengths must match; NaN if no complete pair exists.
double rmse(const Series& actual, const Series& predicted) {
  const auto a = actual.values(), p = predicted.values();
  if (a.size() != p.size()) {
    throw std::invalid_argument("rmse() length mismatch");
  }
  double sum = 0.0;
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].has_value() && p[i].has_value()) {
      const double d = *a[i] - *p[i];
      sum += d * d;
      ++n;
    }
  }
  return n == 0 ? std::numeric_limits<double>::quiet_NaN()
                : std::sqrt(sum / static_cast<double>(n));
}

}  // namespace dsts