// 02_data_science/src/join.cpp
// dsts reshaping: row concatenation, key-based merge (inner/left), mean
// pivot tables and wide-to-long melt.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 05_joins_reshape. Exercised by ctest -R dsts_regression.
// Keys and group labels are compared through DataFrame::cell text, so a
// numeric key 3 and the string "3" are treated as equal.

#include "dsts/join.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace dsts {

// Stack frames vertically. Every frame must have the same column names in
// the same order and matching numeric/string types; otherwise invalid_argument.
DataFrame concat_rows(const std::vector<DataFrame>& frames) {
  if (frames.empty()) return DataFrame();
  const DataFrame& first = frames.front();
  const std::vector<std::string> cols = first.columns();
  for (const DataFrame& f : frames) {
    if (f.columns() != cols) {
      throw std::invalid_argument("concat_rows: column lists must match");
    }
  }
  for (const std::string& c : cols) {
    for (const DataFrame& f : frames) {
      if (f.is_numeric(c) != first.is_numeric(c)) {
        throw std::invalid_argument(
            "concat_rows: column type mismatch for '" + c + "'");
      }
    }
  }
  DataFrame out;
  for (const std::string& c : cols) {
    if (first.is_numeric(c)) {
      std::vector<OptD> v;
      for (const DataFrame& f : frames) {
        const auto& col = f.numeric(c);
        v.insert(v.end(), col.begin(), col.end());
      }
      out.add_numeric(c, std::move(v));
    } else {
      std::vector<std::string> v;
      for (const DataFrame& f : frames) {
        const auto& col = f.strings(c);
        v.insert(v.end(), col.begin(), col.end());
      }
      out.add_string(c, std::move(v));
    }
  }
  return out;
}

// SQL-style join on a single key column.
//   how == "inner": keep left rows whose key exists in right.
//   how == "left":  keep every left row; unmatched right cells are missing.
// Right keys must be unique. Output columns: all left columns, then the right
// columns except the key. Left row order is preserved.
DataFrame merge(const DataFrame& left, const DataFrame& right,
                const std::string& key, const std::string& how) {
  if (!left.has_column(key) || !right.has_column(key)) {
    throw std::invalid_argument("merge: key column '" + key + "' missing");
  }
  if (left.is_numeric(key) != right.is_numeric(key)) {
    throw std::invalid_argument(
        "merge: key column type mismatch for '" + key + "'");
  }
  if (how != "inner" && how != "left") {
    throw std::invalid_argument("merge: how must be \"inner\" or \"left\"");
  }

  // Map every right key to its row (keys must be unique).
  std::map<std::string, size_t> right_index;
  for (size_t r = 0; r < right.rows(); ++r) {
    const std::string k = right.cell(r, key);
    if (right_index.count(k)) {
      throw std::invalid_argument(
          "merge: duplicate key '" + k + "' in right frame");
    }
    right_index.emplace(k, r);
  }

  // Decide which left rows survive.
  std::vector<size_t> kept;
  kept.reserve(left.rows());
  for (size_t r = 0; r < left.rows(); ++r) {
    const bool found = right_index.count(left.cell(r, key)) > 0;
    if (found || how == "left") kept.push_back(r);
  }

  // Collect output columns: left columns, then right columns minus the key.
  std::vector<std::string> out_cols = left.columns();
  for (const std::string& c : right.columns()) {
    if (c != key) out_cols.push_back(c);
  }

  std::map<std::string, std::vector<OptD>> onums;
  std::map<std::string, std::vector<std::string>> ostrs;

  for (const std::string& c : left.columns()) {
    if (left.is_numeric(c)) {
      std::vector<OptD> v;
      v.reserve(kept.size());
      for (size_t r : kept) v.push_back(left.numeric(c)[r]);
      onums.emplace(c, std::move(v));
    } else {
      std::vector<std::string> v;
      v.reserve(kept.size());
      for (size_t r : kept) v.push_back(left.strings(c)[r]);
      ostrs.emplace(c, std::move(v));
    }
  }
  for (const std::string& c : right.columns()) {
    if (c == key) continue;
    const bool numeric = right.is_numeric(c);
    if (numeric) {
      std::vector<OptD> v;
      v.reserve(kept.size());
      for (size_t r : kept) {
        const auto it = right_index.find(left.cell(r, key));
        v.push_back(it != right_index.end() ? right.numeric(c)[it->second]
                                            : OptD());
      }
      onums.emplace(c, std::move(v));
    } else {
      std::vector<std::string> v;
      v.reserve(kept.size());
      for (size_t r : kept) {
        const auto it = right_index.find(left.cell(r, key));
        v.push_back(it != right_index.end() ? right.strings(c)[it->second]
                                            : std::string());
      }
      ostrs.emplace(c, std::move(v));
    }
  }

  DataFrame out;
  for (const std::string& c : out_cols) {
    if (onums.count(c)) {
      out.add_numeric(c, onums.at(c));
    } else {
      out.add_string(c, ostrs.at(c));
    }
  }
  return out;
}

// Mean of `value_col` for each (index_col, columns_col) pair, laid out with one
// row per distinct index value and one numeric column per distinct column value.
// Both orders follow first appearance. Empty cells are missing. Missing values
// are ignored in the mean.
DataFrame pivot_table(const DataFrame& df, const std::string& index_col,
                      const std::string& columns_col,
                      const std::string& value_col) {
  if (!df.has_column(index_col) || !df.has_column(columns_col) ||
      !df.has_column(value_col) || !df.is_numeric(value_col)) {
    throw std::invalid_argument("pivot_table: bad column arguments");
  }

  std::vector<std::string> id_order;
  std::vector<std::string> col_order;
  for (size_t r = 0; r < df.rows(); ++r) {
    const std::string i = df.cell(r, index_col);
    const std::string c = df.cell(r, columns_col);
    if (std::find(id_order.begin(), id_order.end(), i) == id_order.end())
      id_order.push_back(i);
    if (std::find(col_order.begin(), col_order.end(), c) == col_order.end())
      col_order.push_back(c);
  }

  // (index, column) -> (sum, count) for the mean.
  std::map<std::pair<std::string, std::string>, std::pair<double, size_t>> acc;
  const auto& vals = df.numeric(value_col);
  for (size_t r = 0; r < df.rows(); ++r) {
    const OptD& v = vals[r];
    if (!v.has_value()) continue;
    auto& cell = acc[{df.cell(r, index_col), df.cell(r, columns_col)}];
    cell.first += *v;
    cell.second += 1;
  }
  DataFrame out;
  if (df.is_numeric(index_col)) {
    std::vector<OptD> idx(id_order.size(), OptD(0.0));
    // fill real values from the source cell text
    for (size_t k = 0; k < id_order.size(); ++k) {
      for (size_t r = 0; r < df.rows(); ++r) {
        if (df.cell(r, index_col) == id_order[k]) {
          idx[k] = df.numeric(index_col)[r];
          break;
        }
      }
    }
    out.add_numeric(index_col, std::move(idx));
  } else {
    out.add_string(index_col, id_order);
  }
  for (const std::string& c : col_order) {
    std::vector<OptD> col;
    col.reserve(id_order.size());
    for (const std::string& i : id_order) {
      const auto it = acc.find({i, c});
      col.push_back(it != acc.end() ? OptD(it->second.first / it->second.second)
                                    : OptD());
    }
    out.add_numeric(c, std::move(col));
  }
  return out;
}

// Wide to long: for each value column, emit one row per source row with the id
// columns copied, `variable` = column name and `value` = the cell. Rows are
// ordered by value column first, then source row. Value columns must be numeric.
DataFrame melt(const DataFrame& df, const std::vector<std::string>& id_cols,
               const std::vector<std::string>& value_cols) {
  if (id_cols.empty() || value_cols.empty()) {
    throw std::invalid_argument("melt: need id_cols and value_cols");
  }
  for (const std::string& c : id_cols) {
    if (!df.has_column(c)) {
      throw std::invalid_argument("melt: unknown id column '" + c + "'");
    }
  }
  for (const std::string& c : value_cols) {
    if (!df.has_column(c) || !df.is_numeric(c)) {
      throw std::invalid_argument(
          "melt: unknown or non-numeric value column '" + c + "'");
    }
  }

  const size_t n = df.rows() * value_cols.size();
  std::map<std::string, std::vector<OptD>> id_num;
  std::map<std::string, std::vector<std::string>> id_str;
  for (const std::string& c : id_cols) {
    if (df.is_numeric(c)) {
      id_num[c].reserve(n);
    } else {
      id_str[c].reserve(n);
    }
  }
  std::vector<std::string> variable;
  std::vector<OptD> value;
  variable.reserve(n);
  value.reserve(n);

  for (const std::string& vc : value_cols) {
    const auto& src = df.numeric(vc);
    for (size_t r = 0; r < df.rows(); ++r) {
      for (const std::string& c : id_cols) {
        if (df.is_numeric(c)) {
          id_num[c].push_back(df.numeric(c)[r]);
        } else {
          id_str[c].push_back(df.strings(c)[r]);
        }
      }
      variable.push_back(vc);
      value.push_back(src[r]);
    }
  }

  DataFrame out;
  for (const std::string& c : id_cols) {
    if (df.is_numeric(c)) {
      out.add_numeric(c, std::move(id_num.at(c)));
    } else {
      out.add_string(c, std::move(id_str.at(c)));
    }
  }
  out.add_string("variable", std::move(variable));
  out.add_numeric("value", std::move(value));
  return out;
}

}  // namespace dsts