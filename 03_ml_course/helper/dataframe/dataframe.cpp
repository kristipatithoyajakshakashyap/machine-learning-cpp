#include "helper/dataframe/dataframe.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ml {

// ===========================================================================
// Implementation of DataFrame + the tiny CSV reader.
//
// These are the exact parsing rules pandas.read_csv() follows:
//   - cells are split on commas; a quoted "..." group is one cell and may
//     contain commas; doubled "" inside quotes is one literal quote;
//   - leading/trailing whitespace and one pair of surrounding quotes are
//     stripped from each cell;
//   - a column is NUMERIC when every non-empty cell parses as a double;
//     empty cells in a numeric column become missing (nullopt).
//   - otherwise the whole column is a STRING column.
// The behaviour is copied from the 02_data_science dsts toolkit so lessons
// produce bit-identical numbers.
// ===========================================================================

namespace {

// Strip whitespace and surrounding double quotes from one CSV cell.
std::string trim_cell(const std::string& cell) {
  const size_t a = cell.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = cell.find_last_not_of(" \t\r\n");
  std::string s = cell.substr(a, b - a + 1);
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
    const size_t x = s.find_first_not_of(" \t\r\n");
    if (x == std::string::npos) return "";
    const size_t y = s.find_last_not_of(" \t\r\n");
    s = s.substr(x, y - x + 1);
  }
  return s;
}

// Split one CSV line on commas (quote-aware).
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> cells;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        cur += '"';  // doubled double-quote -> one literal quote
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == ',' && !in_quotes) {
      cells.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  cells.push_back(cur);
  return cells;
}

// Parse "24" or "24.5" to a double; false on anything else.
bool try_stod(const std::string& text, double& out) {
  if (text.empty()) return false;
  const std::string s = trim_cell(text);
  if (s.empty()) return false;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) return false;  // nothing consumed
  for (const char* p = end; *p != '\0'; ++p) {
    if (!std::isspace(static_cast<unsigned char>(*p))) return false;
  }
  out = v;
  return true;
}

// True when the whole column's cells are numeric after trimming (empty cells
// are allowed: they become missing).
bool all_cells_numeric(const std::vector<std::string>& cells) {
  for (const auto& raw : cells) {
    const std::string c = trim_cell(raw);
    if (c.empty()) continue;
    double dummy;
    if (!try_stod(c, dummy)) return false;
  }
  return true;
}

std::vector<OptD> to_numeric(const std::vector<std::string>& cells) {
  std::vector<OptD> out;
  out.reserve(cells.size());
  for (const auto& raw : cells) {
    const std::string c = trim_cell(raw);
    double v;
    if (try_stod(c, v)) {
      out.push_back(OptD(v));
    } else {
      out.push_back(OptD());
    }
  }
  return out;
}

std::vector<std::string> to_strings(const std::vector<std::string>& cells) {
  std::vector<std::string> out;
  out.reserve(cells.size());
  for (const auto& raw : cells) out.push_back(trim_cell(raw));
  return out;
}

// Split file text into lines; quoted fields that contain a \n stay intact.
std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> lines;
  std::string cur;
  bool in_quotes = false;
  for (const char ch : text) {
    if (ch == '"') in_quotes = !in_quotes;
    if (ch == '\n' && !in_quotes) {
      lines.push_back(cur);
      cur.clear();
    } else if (ch != '\r') {
      cur += ch;
    }
  }
  if (!cur.empty()) lines.push_back(cur);
  return lines;
}

}  // namespace

// ---------------------------------------------------------------------------
// DataFrame
// ---------------------------------------------------------------------------

bool DataFrame::has_column(const std::string& col) const {
  return nums_.count(col) > 0 || strs_.count(col) > 0;
}

bool DataFrame::is_numeric(const std::string& col) const {
  return nums_.count(col) > 0;
}

const std::vector<OptD>& DataFrame::numeric(const std::string& col) const {
  auto it = nums_.find(col);
  if (it == nums_.end())
    throw std::invalid_argument(
        "DataFrame::numeric: column '" + col +
        "' is not a numeric column");
  return it->second;
}

const std::vector<std::string>& DataFrame::strings(const std::string& col) const {
  auto it = strs_.find(col);
  if (it == strs_.end())
    throw std::invalid_argument(
        "DataFrame::strings: column '" + col +
        "' is not a string column");
  return it->second;
}

void DataFrame::add_numeric(const std::string& name, std::vector<OptD> values) {
  if (rows_ == 0) {
    rows_ = values.size();
  } else if (values.size() != rows_) {
    throw std::invalid_argument(
        "DataFrame::add_numeric: column '" + name + "' has " +
        std::to_string(values.size()) + " cells, expected " +
        std::to_string(rows_));
  }
  if (nums_.count(name) > 0 || strs_.count(name) > 0)
    throw std::invalid_argument(
        "DataFrame::add_numeric: duplicate column '" + name + "'");
  order_.push_back(name);
  nums_.emplace(name, std::move(values));
}

void DataFrame::add_string(const std::string& name, std::vector<std::string> values) {
  if (rows_ == 0) {
    rows_ = values.size();
  } else if (values.size() != rows_) {
    throw std::invalid_argument(
        "DataFrame::add_string: column '" + name + "' has " +
        std::to_string(values.size()) + " cells, expected " +
        std::to_string(rows_));
  }
  if (nums_.count(name) > 0 || strs_.count(name) > 0)
    throw std::invalid_argument(
        "DataFrame::add_string: duplicate column '" + name + "'");
  order_.push_back(name);
  strs_.emplace(name, std::move(values));
}

std::string DataFrame::shape() const {
  return std::to_string(rows_) + " rows x " + std::to_string(order_.size()) +
         " columns";
}

std::string DataFrame::info() const {
  std::ostringstream o;
  o << "DataFrame: " << shape() << "\n";
  for (const auto& name : order_) {
    if (nums_.count(name) > 0) {
      const auto& col = nums_.at(name);
      size_t missing = 0;
      for (const auto& v : col)
        if (!v.has_value()) ++missing;
      o << "  " << name << "  <numeric>  " << col.size() << " cells, "
        << missing << " missing\n";
    } else {
      const auto& col = strs_.at(name);
      o << "  " << name << "  <string>   " << col.size() << " cells\n";
    }
  }
  return o.str();
}

// ---------------------------------------------------------------------------
// CSV reader + number formatting
// ---------------------------------------------------------------------------

DataFrame read_csv(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open file '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const auto text = buffer.str();
  const auto lines = lines_of(text);
  if (lines.empty()) return DataFrame();
  const auto header = split_csv(lines.front());
  DataFrame df;
  for (size_t c = 0; c < header.size(); ++c) {
    std::vector<std::string> col;
    col.reserve(lines.size() - 1);
    for (size_t r = 1; r < lines.size(); ++r) {
      const auto cells = split_csv(lines[r]);
      col.push_back(c < cells.size() ? cells[c] : "");
    }
    if (all_cells_numeric(col)) {
      df.add_numeric(trim_cell(header[c]), to_numeric(col));
    } else {
      df.add_string(trim_cell(header[c]), to_strings(col));
    }
  }
  return df;
}

std::string fmt(double value) {
  std::ostringstream os;
  os << std::setprecision(7) << value;
  std::string s = os.str();
  const size_t dot = s.find('.');
  if (dot != std::string::npos) {
    size_t last = s.find_last_not_of('0');
    if (last > dot) {
      s.erase(last + 1);
    } else {
      s.erase(dot);
    }
  }
  if (s == "-0") s = "0";
  return s;
}

}  // namespace ml