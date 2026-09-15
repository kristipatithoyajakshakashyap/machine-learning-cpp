#pragma once
// 03_ml_course/01_supervised/04_ensembles/01_voting_and_stacking/Model.hpp
//
// Purpose : VotingStacking, a heterogeneous classification ensemble of three
//           course models (multinomial logistic regression, 5-NN, 50-tree
//           random forest) combined by hard voting, soft voting or stacking.
// Inputs  : a finite, rectangular feature matrix X and class codes y in
//           0..K-1 (K >= 2). The pipeline standardises X; this class does not.
// Outputs : none written here; persisted through save()/load() with the tag
//           "VotingStacking_V1" (mode, n_classes, then the four members).
// Run     : header-only; used by ens_theory, ens_math_intuition,
//           ens_implementation, ens_end_to_end, ens_predict and
//           tests/model_test.cpp (interface library ml_ens).
//
// VotingStacking: three heterogeneous base learners (multinomial logistic
// regression, 5-NN, 50-tree random forest) combined in one of three ways.
//   mode 0  hard voting   majority of base labels (ties -> smallest label)
//   mode 1  soft voting   average of base predict_proba
//   mode 2  stacking      out-of-fold base probabilities feed a multinomial
//                         logistic meta-learner (5 stratified folds, seed 42)
// Header-only; the base learners are compiled in ml_clog, ml_cknn and
// ml_tree_models.
#include "01_supervised/01_regression/04_random_forest_regressor/RandomForest.hpp"
#include "01_supervised/02_classification/01_logistic_regression/LogisticRegression.hpp"
#include "01_supervised/02_classification/02_knn_classification/KNN.hpp"
#include "helper/eval/cross_validation.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/optim.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace course {

// Fitted state: n_classes_ (0 until fit), the three base members and, in
// stacking mode only, meta_ (a logistic regression over 3 * K stacked
// probabilities). predict() is argmax of predict_proba() in every mode, so
// hard-voting ties resolve to the smallest class label.
class VotingStacking {
public:
  static constexpr int kHard = 0, kSoft = 1, kStacking = 2;
  static constexpr size_t kBaseLearners = 3, kMetaFolds = 5;
  static constexpr unsigned kMetaSeed = 42;
  static constexpr const char *kTag = "VotingStacking_V1";

  // mode is a double because the pipeline passes grid values as doubles; it
  // must be exactly 0, 1 or 2 (throws std::invalid_argument otherwise).
  explicit VotingStacking(double mode = 0) : mode_(static_cast<int>(mode)) {
    if (mode != std::floor(mode) || mode_ < kHard || mode_ > kStacking)
      throw std::invalid_argument("VotingStacking mode must be 0, 1 or 2");
  }

  // Read-only accessors used by the lessons and the tests.
  int mode() const { return mode_; }
  size_t n_classes() const { return n_classes_; }
  const ml::LogisticRegression &meta() const { return meta_; }
  const ml::LogisticRegression &logistic() const { return logistic_; }
  const ml::KNNClassifier &knn() const { return knn_; }
  const ml::RandomForest &forest() const { return forest_; }

  // Validate, count classes, (stacking only) fit the meta-learner on
  // out-of-fold member probabilities, then refit all members on every row.
  // Throws std::invalid_argument on empty/ragged/non-finite X or labels that
  // are not class codes. Complexity: the three member fits, times
  // (kMetaFolds + 1) in stacking mode.
  void fit(const ml::Mat &X, const ml::Vec &y) {
    validate(X, y);
    n_classes_ = class_count(y);
    if (mode_ == kStacking)
      fit_meta(X, y);
    fit_bases(X, y);
  }

  // Rows of K class probabilities: vote fractions (hard), the mean of the
  // member probabilities (soft) or the meta-learner output (stacking).
  // Throws std::runtime_error when unfitted; an empty X yields an empty Mat.
  ml::Mat predict_proba(const ml::Mat &X) const {
    if (n_classes_ == 0)
      throw std::runtime_error("VotingStacking is not fitted");
    if (X.empty())
      return {};
    if (mode_ == kHard)
      return vote_fractions(X);
    const ml::Mat stacked = base_probabilities(X);
    if (mode_ == kStacking)
      return meta_.predict_proba(stacked);
    // Soft voting: average block b (columns b*K .. b*K+K-1) over the members.
    ml::Mat out(X.size(), ml::Vec(n_classes_, 0.0));
    for (size_t i = 0; i < X.size(); ++i)
      for (size_t b = 0; b < kBaseLearners; ++b)
        for (size_t c = 0; c < n_classes_; ++c)
          out[i][c] += stacked[i][b * n_classes_ + c] /
                       static_cast<double>(kBaseLearners);
    return out;
  }

  // Class code of the largest probability per row (first max wins ties).
  ml::Vec predict(const ml::Mat &X) const {
    ml::Vec out;
    for (const auto &row : predict_proba(X))
      out.push_back(static_cast<double>(
          std::max_element(row.begin(), row.end()) - row.begin()));
    return out;
  }

  // Base-learner probabilities side by side: [lr | knn | forest], each block
  // n_classes wide. This is the meta-learner's input.
  ml::Mat base_probabilities(const ml::Mat &X) const {
    return concat(logistic_.predict_proba(X), knn_.predict_proba(X),
                  forest_.predict_proba(X));
  }

  // Serialise tag, mode, class count and the four members in a fixed order.
  // meta_ is written in every mode (unfitted in voting modes) so the layout
  // is identical and load() needs no branching.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string(kTag), mode_, n_classes_);
    logistic_.save(out);
    knn_.save(out);
    forest_.save(out);
    meta_.save(out);
  }
  // Inverse of save(); rejects a foreign tag, an unknown mode or K < 2.
  void load(std::istream &in) {
    std::string tag;
    ml::archive::read(in, tag);
    if (tag != kTag)
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, mode_, n_classes_);
    if (mode_ < kHard || mode_ > kStacking || n_classes_ < 2)
      throw std::runtime_error("Invalid VotingStacking state");
    logistic_.load(in);
    knn_.load(in);
    forest_.load(in);
    meta_.load(in);
  }

private:
  int mode_;
  size_t n_classes_ = 0;
  ml::LogisticRegression logistic_;
  ml::KNNClassifier knn_{5};
  ml::RandomForest forest_;
  ml::LogisticRegression meta_;

  // Guards for fit(): non-empty, rectangular, finite X with one label per row.
  static void validate(const ml::Mat &X, const ml::Vec &y) {
    if (X.empty() || X[0].empty() || X.size() != y.size())
      throw std::invalid_argument("VotingStacking needs matching X and y");
    for (const auto &row : X) {
      if (row.size() != X[0].size())
        throw std::invalid_argument("Ragged feature matrix");
      for (double v : row)
        if (!std::isfinite(v))
          throw std::invalid_argument("Nonfinite feature");
    }
  }
  // K = max label + 1; labels must be non-negative integers and K >= 2.
  static size_t class_count(const ml::Vec &y) {
    double top = 0;
    for (double v : y) {
      if (!std::isfinite(v) || v < 0 || v != std::floor(v))
        throw std::invalid_argument("Labels must be class codes 0..K-1");
      top = std::max(top, v);
    }
    const size_t k = static_cast<size_t>(top) + 1;
    if (k < 2)
      throw std::invalid_argument("Need at least two classes");
    return k;
  }
  // Row-wise concatenation [a | b | c]; all three must have the same height.
  static ml::Mat concat(const ml::Mat &a, const ml::Mat &b, const ml::Mat &c) {
    ml::Mat out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      out[i] = a[i];
      out[i].insert(out[i].end(), b[i].begin(), b[i].end());
      out[i].insert(out[i].end(), c[i].begin(), c[i].end());
    }
    return out;
  }
  // Member factories: K-class softmax regression (C = 1.0, Adam lr 0.05,
  // 600 epochs) and a 50-tree classification forest (depth 8, min leaf 2).
  static ml::LogisticRegression make_logistic(size_t k) {
    return ml::LogisticRegression(k, 1.0, 0.05, 600);
  }
  static ml::RandomForest make_forest() {
    ml::RandomForest f(50, 0, 8, 2);
    f.set_task(ml::TreeTask::Classification);
    return f;
  }
  // Fit the three members on the rows given (all rows, or a fold's train set).
  void fit_bases(const ml::Mat &X, const ml::Vec &y) {
    logistic_ = make_logistic(n_classes_);
    logistic_.fit(X, y);
    knn_ = ml::KNNClassifier(5);
    knn_.fit(X, y);
    forest_ = make_forest();
    forest_.fit(X, y);
  }
  // Out-of-fold base probabilities: every training row is scored by base
  // models that never saw it, so the meta-learner cannot reward overfitting.
  void fit_meta(const ml::Mat &X, const ml::Vec &y) {
    const auto folds = ml::stratified_kfold(y, kMetaFolds, kMetaSeed);
    ml::Mat oof(X.size(), ml::Vec(kBaseLearners * n_classes_, 0.0));
    for (const auto &fold : folds) {
      ml::Mat Xtr, Xte;
      ml::Vec ytr;
      for (size_t i : fold.train) {
        Xtr.push_back(X[i]);
        ytr.push_back(y[i]);
      }
      for (size_t i : fold.test)
        Xte.push_back(X[i]);
      // A throwaway soft-voting instance trains the members on this fold.
      VotingStacking part(kSoft);
      part.n_classes_ = n_classes_;
      part.fit_bases(Xtr, ytr);
      const ml::Mat probs = part.base_probabilities(Xte);
      for (size_t r = 0; r < fold.test.size(); ++r)
        oof[fold.test[r]] = probs[r];
    }
    meta_ = make_logistic(n_classes_);
    meta_.fit(oof, y);
  }
  // Hard voting as probabilities: each member adds 1/3 to its predicted class.
  ml::Mat vote_fractions(const ml::Mat &X) const {
    const ml::Vec votes[kBaseLearners] = {logistic_.predict(X), knn_.predict(X),
                                          forest_.predict(X)};
    ml::Mat out(X.size(), ml::Vec(n_classes_, 0.0));
    for (size_t i = 0; i < X.size(); ++i)
      for (const auto &v : votes)
        out[i][static_cast<size_t>(v[i])] += 1.0 / kBaseLearners;
    return out;
  }
};

} // namespace course
