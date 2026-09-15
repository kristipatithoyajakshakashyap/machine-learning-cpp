#ifndef HELPER_DATAFRAME_DATAFRAME_HPP
#define HELPER_DATAFRAME_DATAFRAME_HPP

// ===========================================================================
// helper/dataframe/dataframe.hpp
// ---------------------------------------------------------------------------
// The course's own tiny "pandas.DataFrame": a column store with a strict
// rectangular shape.  It is used by every lesson that needs tabular data:
//
//   - helper/data/datasets.cpp  includes this header and turns every CSV into
//     an ml::Dataset via DataFrame::numeric / DataFrame::strings.
//   - Every lesson that talks about data reads a CSV through ml::read_csv.
//     The 01_datasets_overview lesson is the first user.
//
// A column is either NUMERIC (std::vector<OptD>, missing cells = nullopt) or
// STRING (std::vector<std::string>); a column never mixes both types.  The
// tiny CSV parser in dataframe.cpp decides which is which: a column becomes
// numeric only when every non-empty cell parses as a double, exactly like
// pandas.read_csv().
//
// The pandas/numpy equivalent of this file is pandas.DataFrame (yes, this
// whole file is our "pip install pandas").
// ===========================================================================

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ml {

// "Optional double": a numeric cell value, or nullopt when the cell was empty
// (missing data) in the CSV.  Same idea as pandas' NaN.
using OptD = std::optional<double>;

class DataFrame {
 public:
  DataFrame() = default;

  // --- shape / metadata ------------------------------------------------
  bool empty() const { return rows_ == 0; }
  size_t rows() const { return rows_; }
  size_t cols() const { return order_.size(); }
  std::vector<std::string> columns() const { return order_; }
  bool has_column(const std::string& col) const;
  bool is_numeric(const std::string& col) const;  // false if not found

  // --- column access ---------------------------------------------------
  // Throws std::invalid_argument when the column does not exist or its type
  // does not match the accessor (numeric vs strings).
  const std::vector<OptD>& numeric(const std::string& col) const;
  const std::vector<std::string>& strings(const std::string& col) const;

  // --- building ---------------------------------------------------------
  // Every added column must be exactly rows() long (or rows() == 0).
  void add_numeric(const std::string& name, std::vector<OptD> values);
  void add_string(const std::string& name, std::vector<std::string> values);

  // --- printing (used by the EDA lessons) -------------------------------
  std::string shape() const;  // "150 rows x 5 columns"
  std::string info() const;   // per-column name / type / missing-cell count

 private:
  std::vector<std::string> order_;
  std::map<std::string, std::vector<OptD>> nums_;
  std::map<std::string, std::vector<std::string>> strs_;
  size_t rows_ = 0;
};

// Read a CSV file into a DataFrame (pandas.read_csv equivalent).
// Throws std::runtime_error when the file cannot be opened.
DataFrame read_csv(const std::string& path);

// Deterministic number-to-text: 21.0 -> "21", 2.4494897427 -> "2.44949".
std::string fmt(double value);

}  // namespace ml

#endif  // HELPER_DATAFRAME_DATAFRAME_HPP