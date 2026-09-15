// 02_data_science/src/cleaning.cpp
// dsts cleaning utilities: missing-row removal, duplicate detection, IQR
// outlier flags, one-hot encoding and min-max / z-score scaling.
//
// Reads/writes: nothing on disk; pure functions over DataFrame and Series.
// Build:  part of the dsts static library (target dsts); linked into every
//         02_data_science lesson. Exercised by ctest -R dsts_regression.
// Conventions: a numeric cell is missing when its OptD is empty; a string
// cell is missing when it is "". Scalers keep missing cells missing.

#include "dsts/cleaning.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace dsts {

namespace {

// True if the cell (row, col) is missing under the numeric/string convention above.
bool col_cell_missing(const DataFrame& df, size_t row, const std::string& col) {
  if (df.is_numeric(col)) {
    const auto& v = df.numeric(col).at(row);
    return !v.has_value();
  }
  return df.strings(col).at(row).empty();
}

// True if rows a and b agree in every column (compared via DataFrame::cell).
bool row_matches(const DataFrame& df, size_t a, size_t b) {
  for (const auto& c : df.columns()) {
    if (df.cell(a, c) != df.cell(b, c)) return false;
  }
  return true;
}

}  // namespace

// Drop rows with any missing cell. `cols` empty means check every column;
// otherwise only the listed columns are inspected. Row order is preserved.
DataFrame dropna(const DataFrame& df, const std::vector<std::string>& cols) {
  std::vector<size_t> keep;
  for (size_t r = 0; r < df.rows(); ++r) {
    bool ok = true;
    if (cols.empty()) {
      for (const auto& c : df.columns()) {
        if (col_cell_missing(df, r, c)) {
          ok = false;
          break;
        }
      }
    } else {
      for (const auto& c : cols) {
        if (col_cell_missing(df, r, c)) {
          ok = false;
          break;
        }
      }
    }
    if (ok) keep.push_back(r);
  }
  return df.select_rows(keep);
}

// Mark row i as duplicated if an identical row j < i exists (the first
// occurrence is never marked). O(n^2) pairwise comparison; fine for lesson data.
std::vector<bool> duplicated(const DataFrame& df) {
  std::vector<bool> out(df.rows(), false);
  for (size_t i = 1; i < df.rows(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (row_matches(df, i, j)) {
        out[i] = true;
        break;
      }
    }
  }
  return out;
}

// Keep only the first occurrence of each distinct row.
DataFrame drop_duplicates(const DataFrame& df) {
  std::vector<size_t> keep;
  for (size_t i = 0; i < df.rows(); ++i) {
    bool first = true;
    for (size_t j = 0; j < i; ++j) {
      if (row_matches(df, i, j)) {
        first = false;
        break;
      }
    }
    if (first) keep.push_back(i);
  }
  return df.select_rows(keep);
}

// Tukey fences: {Q1 - 1.5*IQR, Q3 + 1.5*IQR}. Missing values are ignored
// by Series::quantile.
std::pair<double, double> iqr_fences(const Series& x) {
  const double q1 = x.quantile(0.25);
  const double q3 = x.quantile(0.75);
  const double i = q3 - q1;
  return {q1 - 1.5 * i, q3 + 1.5 * i};
}

// One flag per cell: true when the value is present and outside the fences.
std::vector<bool> outlier_mask(const Series& x) {
  const auto [lo, hi] = iqr_fences(x);
  std::vector<bool> out;
  out.reserve(x.size());
  const auto values = x.values();
  for (const OptD& v : values) out.push_back(v.has_value() && (*v < lo || *v > hi));
  return out;
}

// Replace string column `col` with one 0/1 numeric column per distinct label,
// named col_label, in order of first appearance. Other columns are copied so the
// overall column order is preserved.
DataFrame one_hot(const DataFrame& df, const std::string& col) {
  const auto& cats = df.strings(col);
  std::vector<std::string> labels;  // first-appearance order
  for (const auto& c : cats) {
    if (std::find(labels.begin(), labels.end(), c) == labels.end()) labels.push_back(c);
  }
  DataFrame out;
  for (const auto& c : df.columns()) {
    if (c == col) {
      for (const auto& label : labels) {
        std::vector<OptD> ind(cats.size(), OptD(0.0));
        for (size_t r = 0; r < cats.size(); ++r) {
          if (cats[r] == label) ind[r] = OptD(1.0);
        }
        out.add_numeric(col + "_" + label, std::move(ind));
      }
    } else if (df.is_numeric(c)) {
      out.add_numeric(c, df.numeric(c));
    } else {
      out.add_string(c, df.strings(c));
    }
  }
  return out;
}

// Scale present values to [0, 1]. A constant series maps to 0 (avoids 0/0).
Series minmax_scale(const Series& x) {
  const double lo = x.min();
  const double hi = x.max();
  std::vector<OptD> out;
  out.reserve(x.size());
  const auto values = x.values();
  for (const OptD& v : values) {
    if (!v.has_value()) {
      out.push_back(OptD{});
    } else if (hi == lo) {
      out.push_back(OptD(0.0));
    } else {
      out.push_back(OptD((*v - lo) / (hi - lo)));
    }
  }
  return Series(x.name(), std::move(out));
}

// Standardise present values to mean 0, population std-dev 1. An all-missing
// series is returned unchanged; a constant series maps to 0.
Series zscore_scale(const Series& x) {
  const auto values = x.values();
  size_t n = 0;
  double m = 0.0;
  for (const OptD& v : values) {
    if (!v.has_value()) continue;
    m += *v;
    ++n;
  }
  if (n == 0) return x;
  m /= static_cast<double>(n);
  double ss = 0.0;
  for (const OptD& v : values) {
    if (!v.has_value()) continue;
    const double d = *v - m;
    ss += d * d;
  }
  const double sd = std::sqrt(ss / static_cast<double>(n));  // population
  std::vector<OptD> out;
  out.reserve(values.size());
  for (const OptD& v : values) {
    if (!v.has_value()) {
      out.push_back(OptD{});
    } else if (sd == 0.0) {
      out.push_back(OptD(0.0));
    } else {
      out.push_back(OptD((*v - m) / sd));
    }
  }
  return Series(x.name(), std::move(out));
}

}  // namespace dsts