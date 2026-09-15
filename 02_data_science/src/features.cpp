// 02_data_science/src/features.cpp
// dsts feature-engineering helpers: fixed-edge binning (cut), quantile edges
// for qcut, sample skewness and a log1p transform.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 06_feature_engineering. Exercised by
// ctest -R dsts_regression.

#include "dsts/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dsts {

// pandas-style bins: first interval takes the minimum edge, otherwise
// intervals are (edges[i], edges[i+1]] (right-closed, left-open).
// Assign each value a label by which interval of `edges` it falls in.
//   edges:  ascending boundaries, size labels.size() + 1 (else invalid_argument).
//   labels: one per interval. Missing values get "NA".
// Values above the last edge fall into the last bin.
std::vector<std::string> cut(const Series& x, const std::vector<double>& edges,
                             const std::vector<std::string>& labels) {
  if (edges.size() != labels.size() + 1) {
    throw std::invalid_argument(
        "cut: expected edges.size() == labels.size() + 1");
  }
  std::vector<std::string> out;
  out.reserve(x.size());
  for (const OptD& v : x.values()) {
    if (!v.has_value()) {
      out.push_back("NA");
      continue;
    }
    size_t bin = labels.size() - 1;
    if (*v <= edges[0]) {
      bin = 0;
    } else {
      for (size_t i = 1; i < edges.size(); ++i) {
        if (*v <= edges[i]) {
          bin = i - 1;
          break;
        }
      }
    }
    out.push_back(labels[bin]);
  }
  return out;
}

// q+1 quantile boundaries (0, 1/q, ..., 1) for equal-frequency binning.
// Feed the result to cut(). q must be >= 2.
std::vector<double> qcut_edges(const Series& x, size_t q) {
  if (q < 2) {
    throw std::invalid_argument("qcut: q must be at least 2");
  }
  std::vector<double> edges;
  edges.reserve(q + 1);
  for (size_t k = 0; k <= q; ++k) {
    edges.push_back(x.quantile(static_cast<double>(k) / static_cast<double>(q)));
  }
  return edges;
}

// Adjusted Fisher-Pearson sample skewness (pandas' default). Returns 0 for
// fewer than 3 present values or a constant series.
double skew(const Series& x) {
  const auto& values = x.values();
  std::vector<double> present;
  present.reserve(values.size());
  for (const OptD& v : values) {
    if (v.has_value()) present.push_back(*v);
  }
  const double n = static_cast<double>(present.size());
  if (n < 3) return 0.0;
  double mean = 0.0;
  for (double v : present) mean += v;
  mean /= n;
  double m2 = 0.0, m3 = 0.0;
  for (double v : present) {
    const double d = v - mean;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n;
  const double s = std::sqrt(m2);
  if (s == 0.0) return 0.0;
  return (n / ((n - 1.0) * (n - 2.0))) * m3 / (s * s * s);
}

// Elementwise log(1 + x); missing stays missing. The result is named
// "log1p(<name>)".
Series log1p(const Series& x) {
  std::vector<OptD> out;
  out.reserve(x.size());
  for (const OptD& v : x.values()) {
    out.push_back(v ? OptD(std::log1p(*v)) : OptD{});
  }
  return Series("log1p(" + x.name() + ")", std::move(out));
}

}  // namespace dsts