// 02_data_science/src/series.cpp
// dsts::Series implementation: a named 1-D column of OptD (optional double).
// A missing cell is an empty optional. Reductions ignore missing values and
// return NaN when nothing is present; elementwise operations keep missing
// cells missing; comparisons treat missing as false.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts). Exercised by ctest -R dsts_regression.

#include "dsts/series.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace dsts {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Sorted copy of the present (non-missing) values.
std::vector<double> present(const std::vector<OptD>& v) {
  std::vector<double> out;
  out.reserve(v.size());
  for (const OptD& x : v) {
    if (x.has_value()) out.push_back(*x);
  }
  return out;
}

// Right-align a row index in a field of width w for printing.
std::string pad_index(int w, size_t i) {
  std::ostringstream os;
  os << std::setw(w) << i;
  return os.str();
}
}  // namespace

// Bounds-checked element access (throws std::out_of_range).
const OptD& Series::at(size_t index) const {
  return values_.at(index);
}

// Number of present values.
size_t Series::count() const { return present(values_).size(); }

// Number of missing values.
size_t Series::missing() const { return values_.size() - count(); }

// Sum of present values; NaN if none.
double Series::sum() const {
  const auto p = present(values_);
  if (p.empty()) return kNaN;
  double s = 0.0;
  for (double x : p) s += x;
  return s;
}

// Mean of present values; NaN if none.
double Series::mean() const {
  const auto p = present(values_);
  if (p.empty()) return kNaN;
  double s = 0.0;
  for (double x : p) s += x;
  return s / static_cast<double>(p.size());
}

// 50th percentile (see quantile).
double Series::median() const { return quantile(0.5); }

// Sample variance (n-1 denominator); NaN with fewer than 2 present values.
double Series::variance() const {
  const auto p = present(values_);
  if (p.size() < 2) return kNaN;
  const double m = mean();
  double s = 0.0;
  for (double x : p) s += (x - m) * (x - m);
  return s / static_cast<double>(p.size() - 1);
}

// Square root of the sample variance.
double Series::stddev() const {
  const double v = variance();
  return std::isnan(v) ? kNaN : std::sqrt(v);
}

// Smallest present value; NaN if none.
double Series::min() const {
  const auto p = present(values_);
  if (p.empty()) return kNaN;
  return *std::min_element(p.begin(), p.end());
}

// Largest present value; NaN if none.
double Series::max() const {
  const auto p = present(values_);
  if (p.empty()) return kNaN;
  return *std::max_element(p.begin(), p.end());
}

// Linear-interpolation quantile (numpy/pandas default): rank = q*(n-1) on
// the sorted present values, interpolate between the neighbours.
// q outside [0,1] throws; NaN if nothing is present.
double Series::quantile(double q) const {
  if (q < 0.0 || q > 1.0) {
    throw std::invalid_argument("quantile() requires q in [0, 1]");
  }
  auto p = present(values_);
  if (p.empty()) return kNaN;
  std::sort(p.begin(), p.end());
  const double rank = q * static_cast<double>(p.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  return p[lo] + (rank - std::floor(rank)) * (p[hi] - p[lo]);
}

// pandas describe(): count, mean, std, min, quartiles, max, plus a name line.
std::string Series::describe() const {
  const auto p = present(values_);
  std::ostringstream os;
  os << "count    " << p.size() << "\n";
  os << "mean     " << fmt(mean()) << "\n";
  os << "std      " << fmt(stddev()) << "\n";
  os << "min      " << fmt(min()) << "\n";
  os << "25%      " << fmt(quantile(0.25)) << "\n";
  os << "50%      " << fmt(quantile(0.50)) << "\n";
  os << "75%      " << fmt(quantile(0.75)) << "\n";
  os << "max      " << fmt(max()) << "\n";
  os << "Name: " << name_ << ", dtype: float64";
  return os.str();
}

// Print every row, or when longer than max_rows the first half, a "..."
// marker and the last half. Index column width fits the largest index.
std::string Series::to_string(size_t max_rows) const {
  std::ostringstream os;
  const int w = static_cast<int>(std::to_string(values_.empty() ? 0
                                                                : values_.size() - 1).size());
  for (size_t i = 0; i < values_.size(); ++i) {
    if (max_rows > 0 && values_.size() > max_rows && i == (max_rows / 2)) {
      os << pad_index(w, i) << "    ...\n";
      continue;
    }
    if (max_rows > 0 && values_.size() > max_rows &&
        i >= (max_rows / 2) && i < values_.size() - (max_rows - max_rows / 2)) {
      continue;
    }
    os << pad_index(w, i) << "    " << to_text(values_[i]) << "\n";
  }
  os << "Name: " << name_ << ", dtype: float64";
  return os.str();
}

// First n rows.
std::string Series::head(size_t n) const {
  std::ostringstream os;
  const int w = static_cast<int>(std::to_string(values_.empty() ? 0
                                                                : values_.size() - 1).size());
  const size_t k = std::min(n, values_.size());
  for (size_t i = 0; i < k; ++i) {
    os << pad_index(w, i) << "    " << to_text(values_[i]) << "\n";
  }
  os << "Name: " << name_ << ", dtype: float64";
  return os.str();
}

// Last n rows.
std::string Series::tail(size_t n) const {
  std::ostringstream os;
  const int w = static_cast<int>(std::to_string(values_.empty() ? 0
                                                                : values_.size() - 1).size());
  const size_t first = values_.size() > n ? values_.size() - n : 0;
  for (size_t i = first; i < values_.size(); ++i) {
    os << pad_index(w, i) << "    " << to_text(values_[i]) << "\n";
  }
  os << "Name: " << name_ << ", dtype: float64";
  return os.str();
}

// Scalar arithmetic; missing stays missing. sub is add of the negation.
Series Series::add(double v) const {
  std::vector<OptD> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x ? OptD(*x + v) : OptD{});
  return Series(name_, std::move(out));
}

// x - v for every present cell.
Series Series::sub(double v) const { return add(-v); }

// x * v for every present cell.
Series Series::mul(double v) const {
  std::vector<OptD> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x ? OptD(*x * v) : OptD{});
  return Series(name_, std::move(out));
}

// x / v for every present cell; v == 0 yields +/-inf or NaN as in plain double.
Series Series::div(double v) const {
  std::vector<OptD> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x ? OptD(*x / v) : OptD{});
  return Series(name_, std::move(out));
}

namespace {
// Elementwise op (0 +, 1 -, 2 *, 3 /) between equal-length series; a missing
// value on either side yields missing. Result keeps the left name.
Series pair_wise(const Series& a, const Series& b, int op) {
  if (a.size() != b.size()) {
    throw std::invalid_argument(
        "series arithmetic: lengths must match (got " +
        std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
  }
  const auto& lhs = a.values();
  const auto& rhs = b.values();
  std::vector<OptD> out;
  out.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!lhs[i] || !rhs[i]) {
      out.push_back(OptD{});
    } else {
      switch (op) {
        case 0: out.push_back(*lhs[i] + *rhs[i]); break;
        case 1: out.push_back(*lhs[i] - *rhs[i]); break;
        case 2: out.push_back(*lhs[i] * *rhs[i]); break;
        default: out.push_back(*lhs[i] / *rhs[i]); break;
      }
    }
  }
  return Series(a.name(), std::move(out));
}
}  // namespace

// Elementwise +, -, *, / between two equal-length series (see pair_wise).
Series Series::add(const Series& other) const { return pair_wise(*this, other, 0); }
Series Series::sub(const Series& other) const { return pair_wise(*this, other, 1); }
Series Series::mul(const Series& other) const { return pair_wise(*this, other, 2); }
Series Series::div(const Series& other) const { return pair_wise(*this, other, 3); }

// Comparison masks for DataFrame::filter; missing compares false.
std::vector<bool> Series::gt(double v) const {
  std::vector<bool> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x.has_value() && *x > v);
  return out;
}

// x >= v mask; missing -> false.
std::vector<bool> Series::ge(double v) const {
  std::vector<bool> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x.has_value() && *x >= v);
  return out;
}

// x < v mask; missing -> false.
std::vector<bool> Series::lt(double v) const {
  std::vector<bool> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x.has_value() && *x < v);
  return out;
}

// x <= v mask; missing -> false.
std::vector<bool> Series::le(double v) const {
  std::vector<bool> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x.has_value() && *x <= v);
  return out;
}

// Copy without the missing cells (length shrinks).
Series Series::drop_missing() const {
  std::vector<OptD> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) {
    if (x.has_value()) out.push_back(x);
  }
  return Series(name_, std::move(out));
}

// Replace every missing cell with v.
Series Series::fill(double v) const {
  std::vector<OptD> out;
  out.reserve(values_.size());
  for (const OptD& x : values_) out.push_back(x.has_value() ? x : OptD(v));
  return Series(name_, std::move(out));
}

}  // namespace dsts