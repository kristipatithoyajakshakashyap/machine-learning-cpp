// 02_data_science/src/csv.cpp
// dsts CSV reader/writer. Reading infers each column's type: if every
// non-empty cell parses as a number the column becomes numeric (empty cells
// -> missing), otherwise it is kept as strings. Writing quotes fields that
// contain separators, quotes or whitespace and prints doubles with
// max_digits10 so they round-trip exactly.
//
// Reads/writes: the file paths passed by the caller (read_csv / write_csv).
// Build:  part of the dsts static library (target dsts). Exercised by
//         ctest -R dsts_regression. Field splitting (split_csv) and number
//         parsing (try_stod) live in dtypes.cpp.

#include "dsts/csv.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <limits>

#include "dsts/dtypes.hpp"

namespace dsts {

namespace {

// True when the whole text is a number (after trimming).
bool all_cells_numeric(const std::vector<std::string>& cells) {
  for (const auto& raw : cells) {
    const std::string c = trim(raw);
    if (c.empty()) continue;  // empty cell == missing, still numeric column
    double dummy;
    if (!try_stod(c, dummy)) return false;
  }
  return true;
}

// Parse each trimmed cell; anything that fails to parse (including "") is missing.
std::vector<OptD> to_numeric(const std::vector<std::string>& cells) {
  std::vector<OptD> out;
  out.reserve(cells.size());
  for (const auto& raw : cells) {
    const std::string c = trim(raw);
    double v;
    if (try_stod(c, v)) {
      out.push_back(OptD(v));
    } else {
      out.push_back(OptD());  // missing
    }
  }
  return out;
}

// Copy raw cells verbatim (no trimming) for a string column.
std::vector<std::string> to_strings(const std::vector<std::string>& cells) {
  std::vector<std::string> out;
  out.reserve(cells.size());
  for (const auto& raw : cells) out.push_back(raw);
  return out;
}

// Split text into records. A newline inside double quotes does not end the
// record; bare \r (Windows line endings) is dropped outside quotes.
// Throws if the text ends inside an open quote.
std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> lines;
  // Line-based split that keeps quoted fields with \n intact.
  std::string cur;
  bool in_quotes = false;
  for (const char ch : text) {
    if (ch == '"') in_quotes = !in_quotes;
    if (ch == '\n' && !in_quotes) {
      lines.push_back(cur);
      cur.clear();
    } else if (ch != '\r' || in_quotes) {
      cur += ch;
    }
  }
  if (in_quotes) throw std::invalid_argument("unterminated CSV quote");
  if (!cur.empty()) lines.push_back(cur);
  return lines;
}

}  // namespace

// Parse CSV text into a DataFrame. First line is the header. Every data row
// must have exactly as many cells as the header, otherwise invalid_argument.
DataFrame read_csv_string(const std::string& text) {
  const auto lines = lines_of(text);
  if (lines.empty()) return DataFrame();
  const auto header = split_csv(lines.front());
  for(size_t r=1;r<lines.size();++r)
    if(split_csv(lines[r]).size()!=header.size())throw std::invalid_argument("CSV row width differs from header");
  DataFrame df;
  for (size_t c = 0; c < header.size(); ++c) {
    std::vector<std::string> col;
    col.reserve(lines.size() - 1);
    for (size_t r = 1; r < lines.size(); ++r) {
      const auto cells = split_csv(lines[r]);
      if (c < cells.size()) {
        col.push_back(cells[c]);
      } else {
        col.push_back("");
      }
    }
    if (all_cells_numeric(col)) {
      df.add_numeric(header[c], to_numeric(col));
    } else {
      df.add_string(header[c], to_strings(col));
    }
  }
  return df;
}

// Read the whole file in binary mode (so \r\n is handled by lines_of) and parse it.
DataFrame read_csv(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open file '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return read_csv_string(buffer.str());
}

// Write header + rows. Missing numeric cells are written as empty fields.
void write_csv(const std::string& path, const DataFrame& df) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("cannot write file '" + path + "'");
  }
  const auto cols = df.columns();
  // RFC 4180 style quoting: wrap in quotes and double any embedded quote.
  auto escaped = [](const std::string& text) {
    if (text.find_first_of(",\"\r\n \t") == std::string::npos) return text;
    std::string quoted = "\"";
    for (char ch : text) { if (ch == '"') quoted += '"'; quoted += ch; }
    return quoted + '"';
  };
  for (size_t c = 0; c < cols.size(); ++c) {
    if (c) out << ",";
    const std::string& name = cols[c];
    out << escaped(name);
  }
  out << "\n";
  for (size_t r = 0; r < df.rows(); ++r) {
    for (size_t c = 0; c < cols.size(); ++c) {
      if (c) out << ",";
      const bool numeric_col = df.is_numeric(cols[c]);
      if (numeric_col) {
        const auto& value = df.numeric(cols[c])[r];
        if (value) out << std::setprecision(std::numeric_limits<double>::max_digits10) << *value;
      } else out << escaped(df.strings(cols[c])[r]);
    }
    out << "\n";
  }
  out.flush();
  if (!out) throw std::runtime_error("failed writing CSV '" + path + "'");
}

}  // namespace dsts
