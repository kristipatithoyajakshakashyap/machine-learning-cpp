#ifndef DSTS_CSV_HPP
#define DSTS_CSV_HPP

// Minimal CSV reader/writer with automatic column typing.
// A column becomes numeric (vector<OptD>) when every non-empty cell
// parses as a double; empty cells become missing (std::nullopt).
// Anything else makes the column a string column.
// The pandas equivalent is `pandas.read_csv` / `DataFrame.to_csv`.

#include <string>

#include "dsts/dataframe.hpp"

namespace dsts {

// Throws std::runtime_error when the file cannot be opened.
DataFrame read_csv(const std::string& path);
// Parse CSV text without touching the filesystem (used by tests/lessons).
DataFrame read_csv_string(const std::string& text);

// Write a DataFrame as CSV. Missing numeric cells become empty cells;
// string cells containing commas or quotes are quoted.
void write_csv(const std::string& path, const DataFrame& df);

}  // namespace dsts

#endif  // DSTS_CSV_HPP