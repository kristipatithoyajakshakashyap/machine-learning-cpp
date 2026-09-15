// 03_ml_course/tests/numerical/compare_predictions.cpp
//
// Purpose : persistence check used by every reload_<module> CTest entry
//           (see tests/CMakeLists.txt and tests/reload.cmake). The workflow
//           writes predictions while the model is still in memory; the
//           predict tool reloads the saved model in a fresh process and
//           predicts the same rows. This program proves both agree.
// Inputs  : argv[1] - <run>/quick/evaluation/predictions.csv written by the
//                     end_to_end workflow (prediction in column index 2).
//           argv[2] - <predict>_results/predictions.csv written by the
//                     predict tool (prediction in column index 1).
//           No COURSE_ROOT / DATA_DIR defines are used.
// Outputs : prints only. Exit 0 when every row matches, 1 otherwise.
// Target  : ml_compare_predictions (invoked by reload.cmake, not a test itself)
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Read one numeric column from a CSV file, skipping the header row.
// Parameters: path   - CSV file to open.
//             column - 0-based index of the column holding the prediction.
// Returns the column as doubles, one per non-empty data row.
// Throws std::runtime_error when the file is missing, a row is too short,
// a value is NaN/inf, or no data rows exist. Complexity: O(file size).
std::vector<double> read_predictions(const std::string &path, size_t column) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read " + path);
  std::string line;
  std::getline(in, line); // header row is discarded
  std::vector<double> values;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream row(line);
    std::string field;
    // Advance to the wanted column; every earlier field is read and dropped.
    for (size_t c = 0; c <= column; ++c)
      if (!std::getline(row, field, ','))
        throw std::runtime_error("Missing prediction column");
    const double value = std::stod(field);
    if (!std::isfinite(value))
      throw std::runtime_error("Nonfinite prediction");
    values.push_back(value);
  }
  if (values.empty())
    throw std::runtime_error("No predictions");
  return values;
}

// Compare the two prediction columns row by row.
// A mismatch is any |expected - replay| above 1e-9 * (1 + |expected|), i.e.
// a mixed absolute/relative tolerance that only allows floating-point noise.
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Expected evaluation and inference CSVs");
    // Workflow CSV: id,truth,prediction,... -> column 2.
    const auto expected = read_predictions(argv[1], 2);
    // Predict-tool CSV: id,prediction -> column 1.
    const auto replay = read_predictions(argv[2], 1);
    // Same number of holdout rows must come back from the reloaded model.
    if (expected.size() != replay.size())
      throw std::runtime_error("Row count changed");
    double max_error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      max_error = std::max(max_error, std::abs(expected[i] - replay[i]));
      // Any real difference means save/load did not reproduce the model.
      if (std::abs(expected[i] - replay[i]) >
          1e-9 * (1 + std::abs(expected[i])))
        throw std::runtime_error("Fresh-process prediction mismatch");
    }
    std::cout << expected.size()
              << " fresh-process predictions matched; max error " << max_error
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
