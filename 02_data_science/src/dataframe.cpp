// 02_data_science/src/dataframe.cpp
// dsts::DataFrame implementation: a column store with two typed maps
// (numeric columns of OptD, string columns of std::string) plus `order_`,
// which remembers insertion order so printing matches pandas. All columns
// share one row count `rows_`.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts). Exercised by ctest -R dsts_regression.

#include "dsts/dataframe.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace dsts {

// True if `col` exists as either a numeric or a string column.
bool DataFrame::has_column(const std::string& col) const {
  return nums_.count(col) == 1 || strs_.count(col) == 1;
}

// True only for numeric columns (string columns return false).
bool DataFrame::is_numeric(const std::string& col) const {
  return nums_.count(col) == 1;
}

// Numeric column by name; throws invalid_argument if absent or not numeric.
const std::vector<OptD>& DataFrame::numeric(const std::string& col) const {
  const auto it = nums_.find(col);
  if (it == nums_.end()) {
    throw std::invalid_argument("column '" + col + "' is not numeric");
  }
  return it->second;
}

// String column by name; throws invalid_argument if absent or not string.
const std::vector<std::string>& DataFrame::strings(const std::string& col) const {
  const auto it = strs_.find(col);
  if (it == strs_.end()) {
    throw std::invalid_argument("column '" + col + "' is not a string column");
  }
  return it->second;
}

// Wrap a numeric column in a Series (copies the values).
Series DataFrame::series(const std::string& col) const {
  return Series(col, numeric(col));
}

// Text form of one cell regardless of type ("NA" for a missing number).
// Used for printing and row comparison.
std::string DataFrame::cell(size_t row, const std::string& col) const {
  const auto n = nums_.find(col);
  if (n != nums_.end()) return to_text(n->second.at(row));
  const auto s = strs_.find(col);
  if (s != strs_.end()) return s->second.at(row);
  throw std::invalid_argument("unknown column '" + col + "'");
}

// Add or replace a numeric column. The first column fixes the row count;
// later columns must match it or invalid_argument is thrown.
void DataFrame::add_numeric(const std::string& name, std::vector<OptD> values) {
  if (rows_ == 0) {
    rows_ = values.size();
  } else if (values.size() != rows_) {
    throw std::invalid_argument("numeric column '" + name +
                                "' length != existing row count");
  }
  // Replacing an existing column keeps its column-order position (pandas
  // df["col"] = ... semantics); new names are appended.
  const auto it = std::find(order_.begin(), order_.end(), name);
  if (it == order_.end()) order_.push_back(name);
  nums_[name] = std::move(values);
}

// Add or replace a string column; same length rules as add_numeric.
void DataFrame::add_string(const std::string& name, std::vector<std::string> values) {
  if (rows_ == 0) {
    rows_ = values.size();
  } else if (values.size() != rows_) {
    throw std::invalid_argument("string column '" + name +
                                "' length != existing row count");
  }
  const auto it = std::find(order_.begin(), order_.end(), name);
  if (it == order_.end()) order_.push_back(name);
  strs_[name] = std::move(values);
}

// New frame with the rows in `index` (any order, duplicates allowed).
// Column order and types are preserved; out-of-range indices throw.
DataFrame DataFrame::select_rows(const std::vector<size_t>& index) const {
  DataFrame out;
  for (const std::string& col : order_) {
    const auto n = nums_.find(col);
    if (n != nums_.end()) {
      std::vector<OptD> v;
      v.reserve(index.size());
      for (size_t i : index) v.push_back(n->second.at(i));
      out.add_numeric(col, std::move(v));
      continue;
    }
    const auto s = strs_.find(col);
    std::vector<std::string> v;
    v.reserve(index.size());
    for (size_t i : index) v.push_back(s->second.at(i));
    out.add_string(col, std::move(v));
  }
  return out;
}

// New frame containing only `cols`, in the order given. Unknown names throw.
DataFrame DataFrame::select_columns(const std::vector<std::string>& cols) const {
  DataFrame out;
  for (const std::string& c : cols) {
    const auto n = nums_.find(c);
    if (n != nums_.end()) {
      out.add_numeric(c, n->second);
      continue;
    }
    const auto s = strs_.find(c);
    if (s != strs_.end()) {
      out.add_string(c, s->second);
      continue;
    }
    throw std::invalid_argument("unknown column '" + c + "'");
  }
  return out;
}

// Keep rows whose mask entry is true. mask.size() must equal rows().
DataFrame DataFrame::filter(const std::vector<bool>& mask) const {
  if (mask.size() != rows_) {
    throw std::invalid_argument("filter() mask length != row count");
  }
  std::vector<size_t> index;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) index.push_back(i);
  }
  return select_rows(index);
}

// Human-readable "R rows x C columns".
std::string DataFrame::shape() const {
  std::ostringstream os;
  os << rows_ << " rows x " << cols() << " columns";
  return os.str();
}

namespace {

// Cap display text at `limit` characters, marking the cut with '~'.
std::string truncate(const std::string& s, size_t limit) {
  if (s.size() <= limit) return s;
  return s.substr(0, limit - 1) + "~";
}

// {0, 1, ..., n-1}
std::vector<size_t> range(size_t n) {
  std::vector<size_t> r;
  r.reserve(n);
  for (size_t i = 0; i < n; ++i) r.push_back(i);
  return r;
}

}  // namespace

// Shared pretty-printer. `rows` are the row indices to show; when max_rows > 0
// and there are more rows than that, the middle is elided (head/tail halves with
// a '.' marker row). Cells are truncated to 16 chars and right-aligned.
std::string DataFrame::render(const std::vector<size_t>& rows,
                              size_t max_rows) const {
  // Which rows actually get printed (with middle elision when truncating).
  std::vector<size_t> shown;
  const bool cut = max_rows > 0 && rows.size() > max_rows;
  if (cut) {
    const size_t head = max_rows / 2;
    const size_t tail = max_rows - head;
    for (size_t i = 0; i < head && i < rows.size(); ++i) shown.push_back(rows[i]);
    if (head + tail < rows.size()) shown.push_back(static_cast<size_t>(-1));
    for (size_t i = rows.size() - tail; i < rows.size(); ++i) shown.push_back(rows[i]);
  } else {
    shown = rows;
  }

  // Per-column display width.
  std::vector<size_t> width;
  for (const std::string& c : order_) {
    size_t w = truncate(c, 16).size();
    for (size_t r : shown) {
      if (r == static_cast<size_t>(-1)) continue;
      w = std::max(w, truncate(cell(r, c), 16).size());
    }
    width.push_back(w);
  }

  // Width of the row-index gutter: digits in the largest shown index.
  const int iw = static_cast<int>(
      std::to_string(rows.empty() ? 0 : rows.back()).size());

  // Strip trailing spaces so lines compare cleanly in tests.
  auto line = [](const std::string& s) {
    size_t end = s.find_last_not_of(' ');
    return end == std::string::npos ? std::string() : s.substr(0, end + 1);
  };

  std::ostringstream os;
  // header (right-aligned, like pandas)
  {
    std::ostringstream hdr;
    hdr << std::setw(iw) << "" << "  ";
    for (size_t j = 0; j < order_.size(); ++j) {
      const std::string h = truncate(order_[j], 16);
      hdr << std::right << std::setw(static_cast<int>(width[j])) << h;
      if (j + 1 < order_.size()) hdr << "  ";
    }
    os << line(hdr.str()) << "\n";
  }
  // body
  for (size_t r : shown) {
    if (r == static_cast<size_t>(-1)) {
      os << std::setw(iw + 2) << "" << std::string(1, '.') << "\n";
      continue;
    }
    std::ostringstream row;
    row << std::right << std::setw(iw) << r << "  ";
    for (size_t j = 0; j < order_.size(); ++j) {
      const std::string& c = order_[j];
      const std::string txt = truncate(cell(r, c), 16);
      row << std::right << std::setw(static_cast<int>(width[j])) << txt;
      if (j + 1 < order_.size()) row << "  ";
    }
    os << line(row.str()) << "\n";
  }
  return os.str();
}

// Whole frame, elided in the middle when longer than max_rows.
std::string DataFrame::to_string(size_t max_rows) const {
  return render(range(rows_), max_rows);
}

// First n rows (or fewer if the frame is smaller), never elided.
std::string DataFrame::head(size_t n) const {
  std::vector<size_t> rows;
  for (size_t i = 0; i < std::min(n, rows_); ++i) rows.push_back(i);
  return render(rows, 0);
}

// Last n rows, never elided.
std::string DataFrame::tail(size_t n) const {
  std::vector<size_t> rows;
  const size_t first = rows_ > n ? rows_ - n : 0;
  for (size_t i = first; i < rows_; ++i) rows.push_back(i);
  return render(rows, 0);
}

// pandas-like info(): one line per column with its dtype and missing count.
std::string DataFrame::info() const {
  std::ostringstream os;
  os << "DataFrame with " << rows_ << " rows x " << cols() << " columns:\n";
  size_t idx = 0;
  for (const std::string& c : order_) {
    const auto n = nums_.find(c);
    if (n != nums_.end()) {
      size_t miss = 0;
      for (const OptD& v : n->second) {
        if (!v.has_value()) ++miss;
      }
      os << "  #" << idx << " " << std::left << std::setw(20) << c
         << "float64   missing=" << miss << "\n";
    } else {
      const auto s = strs_.find(c);
      size_t miss = 0;
      for (const std::string& v : s->second) {
        if (v.empty()) ++miss;
      }
      os << "  #" << idx << " " << std::left << std::setw(20) << c
         << "str       missing=" << miss << "\n";
    }
    ++idx;
  }
  return os.str();
}

}  // namespace dsts