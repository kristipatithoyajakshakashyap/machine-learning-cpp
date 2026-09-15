#ifndef DSTS_DATAFRAME_HPP
#define DSTS_DATAFRAME_HPP

// DataFrame = a column store with a strict rectangular shape.
// Never mixes types inside one column: a column is either numeric
// (vector<OptD>, possibly with missing cells) or string.
// The pandas/numpy equivalent is `pandas.DataFrame`.

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"

namespace dsts {

class DataFrame {
 public:
  DataFrame() = default;

  bool empty() const { return rows_ == 0; }
  size_t rows() const { return rows_; }
  size_t cols() const { return order_.size(); }
  std::vector<std::string> columns() const { return order_; }

  bool has_column(const std::string& col) const;
  bool is_numeric(const std::string& col) const;  // false if not found

  // Direct column access. Throws std::invalid_argument on wrong type.
  const std::vector<OptD>& numeric(const std::string& col) const;
  const std::vector<std::string>& strings(const std::string& col) const;

  // Wrap a numeric column as a Series (its mean/median/... helpers).
  Series series(const std::string& col) const;

  // One cell as text ("NA" for missing numeric cells).
  std::string cell(size_t row, const std::string& col) const;

  // --- building ------------------------------------------------
  // Every added column must be exactly rows() long (or rows() == 0).
  void add_numeric(const std::string& name, std::vector<OptD> values);
  void add_string(const std::string& name, std::vector<std::string> values);

  // --- subsetting ----------------------------------------------
  // Keep the rows listed in index (in that order).
  DataFrame select_rows(const std::vector<size_t>& index) const;
  // Keep the columns whose names appear in cols.
  DataFrame select_columns(const std::vector<std::string>& cols) const;
  // Keep rows where mask[i] == true; mask must be rows() long.
  DataFrame filter(const std::vector<bool>& mask) const;

  // --- printing ------------------------------------------------
  std::string shape() const;  // "150 rows x 5 columns"
  std::string head(size_t n = 5) const;
  std::string tail(size_t n = 5) const;
  // Aligned table over the first/last rows when rows() > max_rows.
  std::string to_string(size_t max_rows = 10) const;
  // Per-column name / type / missing-cell counts.
  std::string info() const;

 private:
  std::vector<std::string> order_;
  std::map<std::string, std::vector<OptD>> nums_;
  std::map<std::string, std::vector<std::string>> strs_;
  size_t rows_ = 0;

  std::string render(const std::vector<size_t>& rows, size_t max_rows) const;
};

}  // namespace dsts

#endif  // DSTS_DATAFRAME_HPP