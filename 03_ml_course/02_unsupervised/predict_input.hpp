// 03_ml_course/02_unsupervised/predict_input.hpp
// Purpose: shared I/O for every `<prefix>_predict` program in the unsupervised
//          and anomaly-detection groups: parse a header-less numeric CSV of new
//          rows and write the model outputs as new_predictions.csv.
// Inputs:  a CSV path supplied on the predict command line (via read_numeric_rows).
// Outputs: results/predict_results/new_predictions.csv (via write_predictions).
// Run target: not an executable; included by ukm_predict, ugm_predict,
//             upca_predict, uif_predict, ulof_predict and uocsvm_predict.
#pragma once
#include <fstream>

#include "workflow.hpp"
namespace ml {
namespace unsup {
// Parse a header-less CSV into a row-major matrix.
// Parameters: path - file to read. Each non-empty line becomes one row.
// Returns: Mat with one Vec per line; empty fields, "NA" and "nan" become NaN so
//          the saved Preprocessor can median-impute them exactly as in training.
// Throws: runtime_error if the file cannot be opened, invalid_argument if a field
//         is not numeric (typically a header row) or the file has no rows.
// Complexity O(total characters).
inline Mat read_numeric_rows(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open input CSV");
  Mat X;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream row(line);
    std::string field;
    Vec values;
    while (std::getline(row, field, ',')) {
      if (field.empty() || field == "NA" || field == "nan")
        values.push_back(std::numeric_limits<double>::quiet_NaN());
      else {
        size_t end = 0;
        double v = std::stod(field, &end);
        // Trailing junk after the number (other than whitespace/CR) means the
        // field was text, e.g. "bill_length_mm" from a header line.
        if (field.find_first_not_of(" \r\t", end) != std::string::npos)
          throw std::invalid_argument("numeric CSV without a header required");
        values.push_back(v);
      }
    }
    // getline drops a trailing empty field ("1,2," -> two fields); restore it.
    if (!line.empty() && line.back() == ',')
      values.push_back(std::numeric_limits<double>::quiet_NaN());
    X.push_back(values);
  }
  if (X.empty()) throw std::invalid_argument("no input rows");
  return X;
}
// Write model outputs as CSV: header "row_id,output_0,...", one line per row.
// Parameters: values - one output vector per input row (all the same width);
//             a - artifact root (RUN_OUTPUT_DIR of the predict target).
// Full 17-digit precision so a reload reproduces the numbers exactly.
inline void write_predictions(const Mat& values, Artifacts& a) {
  std::ostringstream out;
  out << std::setprecision(17) << "row_id";
  for (size_t j = 0; j < values.at(0).size(); ++j) out << ",output_" << j;
  out << '\n';
  for (size_t i = 0; i < values.size(); ++i) {
    out << i;
    for (double v : values[i]) out << ',' << v;
    out << '\n';
  }
  a.write("new_predictions.csv", out.str());
}
}  // namespace unsup
}  // namespace ml
