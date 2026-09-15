#ifndef DSTS_SERIES_HPP
#define DSTS_SERIES_HPP

// Series = one named column of numeric data with missing cells.
// The pandas/numpy equivalent of this file is `pandas.Series`.
// Missing values are stored as std::nullopt and never enter the maths.

#include <string>
#include <vector>

#include "dsts/dtypes.hpp"

namespace dsts {

class Series {
 public:
  Series() = default;
  Series(std::string name, std::vector<OptD> values)
      : name_(std::move(name)), values_(std::move(values)) {}

  const std::string& name() const { return name_; }
  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }
  std::vector<OptD> values() const { return values_; }

  // Index access. Out-of-range throws std::out_of_range.
  const OptD& at(size_t index) const;

  // --- missing-data bookkeeping ---------------------------------------
  size_t count() const;    // non-missing cells
  size_t missing() const;  // missing cells (std::nullopt)

  // --- descriptive statistics (missing cells are ignored) -------------
  double sum() const;
  double mean() const;
  double median() const;
  double variance() const;  // sample variance, divisor n-1
  double stddev() const;    // sample standard deviation
  double min() const;
  double max() const;
  // Linear-interpolation quantile (methods 6-9 style used by pandas:
  // type 7 / "inclusive" default).
  double quantile(double q) const;

  // --- printing -------------------------------------------------------
  // count / mean / std / min / 25% / 50% / 75% / max, pandas-style block.
  std::string describe() const;
  std::string head(size_t n = 5) const;
  std::string tail(size_t n = 5) const;
  // Full display with the column name + dtype footer.
  std::string to_string(size_t max_rows = 10) const;

  // --- elementwise arithmetic (missing propagates) --------------------
  Series add(double v) const;
  Series sub(double v) const;
  Series mul(double v) const;
  Series div(double v) const;

  // Elementwise arithmetic against another Series of the same length;
  // missing cells propagate. Used by module 06 (feature engineering).
  Series add(const Series& other) const;
  Series sub(const Series& other) const;
  Series mul(const Series& other) const;
  Series div(const Series& other) const;

  // --- comparison masks (missing compares as false) -------------------
  std::vector<bool> gt(double v) const;
  std::vector<bool> ge(double v) const;
  std::vector<bool> lt(double v) const;
  std::vector<bool> le(double v) const;

  // --- cleaning helpers used from module 02 ---------------------------
  Series drop_missing() const;  // keep only non-missing cells
  Series fill(double v) const;  // replace missing cells with v

 private:
  std::string name_;
  std::vector<OptD> values_;
};

}  // namespace dsts

#endif  // DSTS_SERIES_HPP