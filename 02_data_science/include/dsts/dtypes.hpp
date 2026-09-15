#ifndef DSTS_DTYPES_HPP
#define DSTS_DTYPES_HPP

// Shared low-level helpers for the dsts toolkit.
// - OptD: an optional double. A missing data cell is std::nullopt.
// - fmt / to_text: deterministic number-to-text printing (no "21.000000").
// - split_csv / trim: the tiny CSV line parser everything else reuses.

#include <optional>
#include <string>
#include <vector>

namespace dsts {

using OptD = std::optional<double>;

// Print a double with trailing zeros trimmed: 21.0 -> "21", 2.4494897...
// -> "2.44949".
std::string fmt(double value);

// "maybe" format: value -> its text, missing -> "NA".
std::string to_text(const OptD& value);

// Strip whitespace and surrounding double quotes from one CSV cell.
std::string trim(const std::string& cell);

// Split one CSV line on commas. Quote-aware: commas and doubled quote
// escapes inside "..." stay intact, so names like "Braund, Mr. Owen
// Harris" parse as a single cell.
std::vector<std::string> split_csv(const std::string& line);

// Does the whole string parse as a double?
bool looks_numeric(const std::string& text);

// Parse "24" or "24.5" to a double (throws nothing; false on failure).
bool try_stod(const std::string& text, double& out);

}  // namespace dsts

#endif  // DSTS_DTYPES_HPP