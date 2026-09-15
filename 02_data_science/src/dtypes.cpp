// 02_data_science/src/dtypes.cpp
// dsts low-level cell helpers shared by CSV parsing and printing: number
// formatting, missing-value text, whitespace/quote trimming, CSV field
// splitting and strict string-to-double conversion.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts). Exercised by ctest -R dsts_regression.

#include "dsts/dtypes.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace dsts {

// Compact display form: 7 significant digits, trailing zeros and a bare
// decimal point removed, and "-0" normalised to "0".
std::string fmt(double value) {
  std::ostringstream os;
  os << std::setprecision(7) << value;
  std::string s = os.str();
  // Trim trailing zeros after the decimal point: 21.0000000 -> "21".
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

// Cell text for a possibly-missing number: fmt(value) or "NA".
std::string to_text(const OptD& value) {
  return value.has_value() ? fmt(*value) : "NA";
}

// Strip surrounding whitespace, then one pair of enclosing double quotes,
// then whitespace again (so "\" 42 \"" becomes "42").
std::string trim(const std::string& cell) {
  // 1. strip leading/trailing whitespace
  const size_t a = cell.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = cell.find_last_not_of(" \t\r\n");
  std::string s = cell.substr(a, b - a + 1);
  // 2. strip one pair of surrounding double quotes
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
    const size_t x = s.find_first_not_of(" \t\r\n");
    if (x == std::string::npos) return "";
    const size_t y = s.find_last_not_of(" \t\r\n");
    s = s.substr(x, y - x + 1);
  }
  return s;
}

// Split one CSV record on commas, honouring double-quoted fields. Quotes
// are removed; a doubled quote inside quotes is one literal quote. Always
// returns at least one cell (an empty line -> {""}).
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> cells;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        // doubled double-quote inside a quoted field -> one literal quote.
        cur += '"';
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

// True if try_stod would succeed on `text`.
bool looks_numeric(const std::string& text) {
  double dummy;
  return try_stod(text, dummy);
}

// Strict parse: the trimmed text must be consumed entirely by strtod apart
// from trailing whitespace. Empty or blank text is not a number.
// On success `out` holds the value; on failure it is left untouched.
bool try_stod(const std::string& text, double& out) {
  if (text.empty()) return false;
  const std::string s = trim(text);
  if (s.empty()) return false;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) return false;  // nothing consumed
  // Everything left over must be trailing whitespace.
  for (const char* p = end; *p != '\0'; ++p) {
    if (!std::isspace(static_cast<unsigned char>(*p))) return false;
  }
  out = v;
  return true;
}

}  // namespace dsts