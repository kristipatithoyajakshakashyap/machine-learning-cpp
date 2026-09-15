#pragma once
// 03_ml_course/01_supervised/03_model_diagnostics/03_hyperparameter_search/Model.hpp
//
// Purpose : TunedForest, the course random forest exposed through ONE scalar
//           parameter so the shared pipeline (which tunes a 1-D grid) can run
//           an exhaustive 2-D search over (max_depth, min_leaf).
// Inputs  : p in {0, ..., 8} at construction; fit(X, y) then takes any
//           feature matrix and class labels accepted by ml::RandomForest.
// Outputs : none written here; persisted through save()/load() with the tag
//           "TunedForest_V1" (index, then the forest archive).
// Run     : header-only; used by hps_implementation, hps_end_to_end,
//           hps_predict and tests/model_test.cpp (library ml_tree_models).
//
// TunedForest: the course random forest exposed through ONE scalar
// parameter so the shared pipeline (a 1-D grid) can drive a 2-D search.
// p is an index into the fixed grid max_depth {4, 8, 16} x min_leaf {1, 2, 5}
// (depth varies fastest), i.e. nine combinations: index 0..8.
#include "01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/Forest.hpp"
#include "helper/persistence/archive.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace course {

// Thin wrapper: index_ (0..8) decodes to depth = kDepths[index % 3] and
// min_leaf = kLeaves[index / 3]; forest_ is built once from that pair.
// Invariant: index_ < grid_size() at all times (checked on every entry).
class TunedForest {
public:
  static constexpr std::array<size_t, 3> kDepths{4, 8, 16};
  static constexpr std::array<size_t, 3> kLeaves{1, 2, 5};
  static constexpr size_t kTrees = 100;
  // Number of grid cells (9) and the two decoders used by the lessons.
  static size_t grid_size() { return kDepths.size() * kLeaves.size(); }
  static size_t depth_of(size_t index) { return kDepths[check(index) % 3]; }
  static size_t leaf_of(size_t index) { return kLeaves[check(index) / 3]; }

  // p must be a non-negative integer below 9 (a double because the pipeline
  // passes grid values as doubles); throws std::invalid_argument otherwise.
  explicit TunedForest(double p = 0.0)
      : index_(decode(p)),
        forest_(make_forest(kTrees, depth_of(index_), leaf_of(index_))) {}

  // Read-only accessors: the chosen grid cell and its decoded (depth, leaf).
  size_t index() const { return index_; }
  size_t max_depth() const { return depth_of(index_); }
  size_t min_leaf() const { return leaf_of(index_); }

  // Forward straight to the 100-tree classification forest.
  void fit(const ml::Mat &X, const ml::Vec &y) { forest_.fit(X, y); }
  ml::Vec predict(const ml::Mat &X) const { return forest_.predict(X); }
  ml::Mat predict_proba(const ml::Mat &X) const {
    return forest_.predict_proba(X);
  }

  // Serialise tag, grid index and the forest; load() re-validates the index
  // so a corrupt archive cannot address the grid out of range.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("TunedForest_V1"), index_, forest_);
  }
  void load(std::istream &in) {
    std::string tag;
    ml::archive::read(in, tag);
    if (tag != "TunedForest_V1")
      throw std::runtime_error("Wrong model type: " + tag);
    ml::archive::read(in, index_, forest_);
    check(index_);
  }

private:
  // Range guard shared by the decoders and load(); returns index unchanged.
  static size_t check(size_t index) {
    if (index >= 9)
      throw std::invalid_argument("TunedForest index must be in [0, 9)");
    return index;
  }
  // Reject NaN/inf, negatives and fractional values before the size_t cast.
  static size_t decode(double p) {
    if (!std::isfinite(p) || p < 0.0 || std::floor(p) != p)
      throw std::invalid_argument("TunedForest parameter must be an integer");
    return check(static_cast<size_t>(p));
  }
  size_t index_;
  ml::RandomForest forest_;
};

} // namespace course
