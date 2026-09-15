#pragma once
// 03_ml_course/01_supervised/03_model_diagnostics/02_feature_selection/Model.hpp
//
// Purpose : SelectedForest, a filter-based feature selector fused with the
//           course random forest so the shared pipeline can tune "how many
//           features to keep" (k) like any other scalar hyper-parameter.
// Inputs  : a feature matrix X (rows x p) and labels y. Classification uses
//           ANOVA F per column, regression |Pearson r|; both from
//           helper/eval/feature_selection.hpp.
// Outputs : none written here; persisted through save()/load() with the tag
//           "SelectedForest_V1" (selector state, then the forest).
// Run     : header-only; used by fsel_implementation, fsel_end_to_end,
//           fsel_predict and tests/model_test.cpp (library ml_fsel).
//
// SelectedForest: filter-based feature selection fused with the course random
// forest. fit() scores every column with ANOVA F on the training rows, keeps
// the k best, and trains a forest on that subset; predict() projects new rows
// onto the same columns. Persisted with tag "SelectedForest_V1".
#include "01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/Forest.hpp"
#include "helper/eval/feature_selection.hpp"
#include "helper/persistence/archive.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace course {

// Selector + forest. Fitted state: scores_ (one filter score per original
// column), kept_ (the k column indices with the highest scores, best first),
// n_features_ (width every later row must have) and the 50-tree forest.
// Invariant after fit(): kept_.size() == k_ <= n_features_ == scores_.size().
class SelectedForest {
public:
  static constexpr const char *kTag = "SelectedForest_V1";
  static constexpr size_t kTrees = 50;

  // k is a double because the pipeline passes grid values as doubles; it must
  // be a positive integer. classification picks ANOVA F (true) or |r| (false).
  explicit SelectedForest(double k = 10, bool classification = true)
      : k_(static_cast<size_t>(k)), classification_(classification) {
    if (!(k >= 1) || k != std::floor(k))
      throw std::invalid_argument("SelectedForest k must be a positive integer");
  }

  // Read-only accessors used by the lessons and the tests.
  size_t k() const { return k_; }
  const std::vector<size_t> &kept() const { return kept_; }
  const ml::Vec &scores() const { return scores_; }
  const ml::RandomForest &forest() const { return forest_; }

  // Score every column on (X, y), keep the top k, fit the forest on the
  // projected matrix. Throws std::invalid_argument if k exceeds the number of
  // columns or the data is empty/ragged/non-finite (checked by filter_scores).
  // Complexity: O(n p) for the scores plus the forest fit on n x k.
  void fit(const ml::Mat &X, const ml::Vec &y) {
    scores_ = ml::filter_scores(X, y, classification_);
    if (k_ > scores_.size())
      throw std::invalid_argument("SelectedForest k exceeds feature count");
    kept_ = ml::top_k(scores_, k_);
    n_features_ = scores_.size();
    forest_ = course::make_forest(kTrees);
    if (!classification_)
      forest_.set_task(ml::TreeTask::Regression);
    forest_.fit(ml::detail::project(X, kept_), y);
  }

  // Predictions on full-width rows; select() drops the unused columns first.
  ml::Vec predict(const ml::Mat &X) const {
    return forest_.predict(select(X));
  }
  // Class probabilities; only meaningful (and allowed) in classification mode.
  ml::Mat predict_proba(const ml::Mat &X) const {
    if (!classification_)
      throw std::runtime_error("predict_proba is classification only");
    return forest_.predict_proba(select(X));
  }

  // Serialise tag, selector state and then the forest's own archive.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string(kTag), k_, classification_,
                       n_features_, kept_, scores_);
    forest_.save(out);
  }
  // Inverse of save(); rejects a foreign tag and any inconsistent kept set
  // (wrong size or an index outside the original feature range).
  void load(std::istream &in) {
    std::string tag;
    ml::archive::read(in, tag);
    if (tag != kTag)
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, k_, classification_, n_features_, kept_, scores_);
    if (k_ == 0 || kept_.size() != k_ || n_features_ < k_)
      throw std::runtime_error("Invalid SelectedForest state");
    for (size_t j : kept_)
      if (j >= n_features_)
        throw std::runtime_error("Kept feature index out of range");
    forest_.load(in);
  }

private:
  size_t k_;
  bool classification_;
  size_t n_features_ = 0;
  std::vector<size_t> kept_;
  ml::Vec scores_;
  ml::RandomForest forest_;

  // Project X onto the kept columns. Throws std::runtime_error when unfitted
  // and std::invalid_argument when a row does not have n_features_ entries.
  ml::Mat select(const ml::Mat &X) const {
    if (kept_.empty())
      throw std::runtime_error("SelectedForest is not fitted");
    for (const auto &row : X)
      if (row.size() != n_features_)
        throw std::invalid_argument("Feature count mismatch");
    return ml::detail::project(X, kept_);
  }
};

} // namespace course
