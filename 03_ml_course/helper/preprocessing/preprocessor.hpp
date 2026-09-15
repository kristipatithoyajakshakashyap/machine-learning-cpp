// 03_ml_course/helper/preprocessing/preprocessor.hpp
//
// Purpose : the shared dense-feature preprocessor: median imputation of
//           NaN cells followed by z-scoring with population statistics.
// Inputs  : a dense feature matrix (ml::Mat) via fit()/transform(); no files.
// Outputs : none directly. run_supervised() archives a fitted instance as
//           results/<stem>_results/<run>/model/preprocessing_state.txt.
// Used by : helper/pipeline/supervised.hpp (fitted inside every CV fold and
//           once on the full training partition), the *_predict inference
//           path, and tests/numerical/core.cpp (part of ml_core, no target).
//
// Contract other modules rely on:
//   * fit() must only ever see training rows; transform() is then applied
//     to validation/test/inference rows so no test statistics leak.
//   * transform() on a NaN cell substitutes the training median of that
//     column; +/-inf is always rejected.
//   * Constant columns get scale 1 (not 0), so they map to 0 instead of NaN.
//   * save()/load() use the archive format; load() re-validates every value.
#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ml {
// Fit median imputation and population scaling on a training partition only.
// Invariants after fit(): median, mean and scale have one entry per feature,
// every entry is finite and scale[j] > 0.
class Preprocessor {
public:
  Vec median, mean, scale;
  // Learn per-column median (from finite values), then mean and population
  // standard deviation computed after imputing NaNs with that median.
  // Parameters: X - training rows (non-empty, rectangular).
  // Throws std::invalid_argument on empty/ragged input, any infinite cell or
  // a column with no finite value. Complexity: O(n * p * log n).
  void fit(const Mat &X) {
    if (X.empty() || X.front().empty())
      throw std::invalid_argument("Preprocessor needs rows and features");
    const size_t p = X.front().size();
    for (const auto &r : X)
      if (r.size() != p)
        throw std::invalid_argument("Ragged feature matrix");
    median.assign(p, 0);
    mean.assign(p, 0);
    scale.assign(p, 1);
    for (size_t j = 0; j < p; ++j) {
      Vec v;
      for (const auto &r : X) {
        if (std::isinf(r[j]))
          throw std::invalid_argument("Infinite feature");
        if (std::isfinite(r[j]))
          v.push_back(r[j]);
      }
      if (v.empty())
        throw std::invalid_argument("Entirely missing training feature");
      std::sort(v.begin(), v.end());
      // Average of the two middle elements handles both odd and even counts.
      median[j] = (v[(v.size() - 1) / 2] + v[v.size() / 2]) / 2;
      for (const auto &r : X)
        mean[j] += std::isnan(r[j]) ? median[j] : r[j];
      mean[j] /= static_cast<double>(X.size());
      double ss = 0;
      for (const auto &r : X) {
        double d = (std::isnan(r[j]) ? median[j] : r[j]) - mean[j];
        ss += d * d;
      }
      scale[j] = std::sqrt(ss / static_cast<double>(X.size()));
      // Constant column: keep scale 1 so the column becomes all zeros.
      if (scale[j] < 1e-12)
        scale[j] = 1;
    }
  }
  // Impute and standardise rows with the fitted statistics.
  // Parameters: X - rows with the same feature count as fit() saw.
  // Returns a new matrix; the input is not modified.
  // Throws std::runtime_error if not fitted, std::invalid_argument on a
  // feature-count mismatch or an infinite cell. Complexity: O(n * p).
  Mat transform(const Mat &X) const {
    if (mean.empty() || median.size() != mean.size() ||
        scale.size() != mean.size())
      throw std::runtime_error("Preprocessor not fitted");
    Mat result = X;
    for (auto &row : result) {
      if (row.size() != mean.size())
        throw std::invalid_argument("Preprocessor feature count mismatch");
      for (size_t j = 0; j < row.size(); ++j) {
        if (std::isinf(row[j]) || !(scale[j] > 0))
          throw std::invalid_argument("Invalid preprocessing value");
        row[j] =
            ((std::isnan(row[j]) ? median[j] : row[j]) - mean[j]) / scale[j];
      }
    }
    return result;
  }
  // Archive the three statistic vectors (see archive.hpp for the format).
  void save(std::ostream &out) const {
    archive::write(out, median, mean, scale);
  }
  // Restore and validate: sizes must agree, every value must be finite and
  // every scale positive, otherwise the archive is rejected.
  void load(std::istream &in) {
    archive::read(in, median, mean, scale);
    if (mean.empty() || median.size() != mean.size() ||
        scale.size() != mean.size())
      throw std::runtime_error("Invalid preprocessing archive");
    for (size_t j = 0; j < mean.size(); ++j)
      if (!std::isfinite(median[j]) || !std::isfinite(mean[j]) ||
          !std::isfinite(scale[j]) || scale[j] <= 0)
        throw std::runtime_error("Invalid saved preprocessing parameters");
  }
};
} // namespace ml
