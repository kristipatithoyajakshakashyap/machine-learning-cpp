#ifndef HELPER_EVAL_DIAGNOSTICS_HPP
#define HELPER_EVAL_DIAGNOSTICS_HPP

// ===========================================================================
// helper/eval/diagnostics.hpp  (header-only)
// ---------------------------------------------------------------------------
// Model diagnostics that work with any fitted model exposing
// Vec predict(const Mat&) const, and any factory returning a model with
// fit(X, y):
//   permutation_importance  - sklearn.inspection.permutation_importance
//   learning_curve          - sklearn.model_selection.learning_curve
// Both are deterministic: permutations and subsampling use a local
// std::mt19937 seeded by the caller, and the shared course RNG is re-seeded
// before every fit so stochastic models (forests) replay exactly.
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "helper/eval/cross_validation.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

namespace ml {

struct ImportanceRow {
  double mean_drop = 0.0; // average metric decrease when the column is shuffled
  double std_drop = 0.0;  // population standard deviation over repeats
};

struct LearningCurvePoint {
  size_t n_train = 0;      // rows used for training at this fraction
  double train_score = 0.0; // metric on the rows the model was fitted on
  double cv_score = 0.0;    // metric on the held-out fold, averaged over folds
};

// metric(y_true, y_pred) -> double, higher is better.
template <class Model, class Metric>
std::vector<ImportanceRow>
permutation_importance(const Model &model, const Mat &X, const Vec &y,
                       Metric metric, size_t repeats, unsigned seed) {
  if (X.empty() || X[0].empty() || X.size() != y.size() || repeats == 0)
    throw std::invalid_argument(
        "permutation_importance needs data, labels and repeats > 0");
  const size_t n = X.size(), p = X[0].size();
  const double base = metric(y, model.predict(X));
  std::mt19937 rng(seed);
  std::vector<ImportanceRow> out(p);
  std::vector<size_t> order(n);
  for (size_t j = 0; j < p; ++j) {
    Vec drops;
    drops.reserve(repeats);
    for (size_t r = 0; r < repeats; ++r) {
      for (size_t i = 0; i < n; ++i)
        order[i] = i;
      std::shuffle(order.begin(), order.end(), rng);
      Mat shuffled = X;
      for (size_t i = 0; i < n; ++i)
        shuffled[i][j] = X[order[i]][j];
      drops.push_back(base - metric(y, model.predict(shuffled)));
    }
    const double m = mean(drops);
    double var = 0.0;
    for (double d : drops)
      var += (d - m) * (d - m);
    out[j].mean_drop = m;
    out[j].std_drop = std::sqrt(var / static_cast<double>(repeats));
  }
  return out;
}

namespace detail {
// First ceil(fraction * count) rows of a seeded shuffle; when stratified,
// the quota is applied per class so every class survives small fractions.
inline std::vector<size_t> subset_rows(const std::vector<size_t> &rows,
                                       const Vec &y, double fraction,
                                       bool stratified, std::mt19937 &rng) {
  std::map<double, std::vector<size_t>> groups;
  for (size_t i : rows)
    groups[stratified ? y[i] : 0.0].push_back(i);
  std::vector<size_t> out;
  for (auto &kv : groups) {
    auto &ids = kv.second;
    std::shuffle(ids.begin(), ids.end(), rng);
    const size_t take = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(fraction * ids.size())));
    out.insert(out.end(), ids.begin(),
               ids.begin() + static_cast<ptrdiff_t>(std::min(take, ids.size())));
  }
  std::sort(out.begin(), out.end());
  return out;
}
} // namespace detail

// make() -> model with fit(X, y) and predict(X). fractions are in (0, 1].
template <class Factory, class Metric>
std::vector<LearningCurvePoint>
learning_curve(Factory make, const Mat &X, const Vec &y,
               const std::vector<double> &fractions, size_t k, unsigned seed,
               Metric metric, bool stratified) {
  if (X.empty() || X.size() != y.size() || fractions.empty() || k < 2)
    throw std::invalid_argument(
        "learning_curve needs data, labels, fractions and k >= 2");
  for (double f : fractions)
    if (!(f > 0.0) || f > 1.0)
      throw std::invalid_argument("learning_curve fractions must be in (0, 1]");
  const auto folds = stratified ? stratified_kfold(y, k, seed) : kfold(X.size(), k, seed);
  std::vector<LearningCurvePoint> out;
  for (double f : fractions) {
    std::mt19937 rng(seed);
    LearningCurvePoint point;
    for (const auto &fold : folds) {
      const auto rows = detail::subset_rows(fold.train, y, f, stratified, rng);
      Mat Xtr, Xte;
      Vec ytr, yte;
      for (size_t i : rows) {
        Xtr.push_back(X[i]);
        ytr.push_back(y[i]);
      }
      for (size_t i : fold.test) {
        Xte.push_back(X[i]);
        yte.push_back(y[i]);
      }
      seed_rng(seed);
      auto model = make();
      model.fit(Xtr, ytr);
      point.n_train += rows.size();
      point.train_score += metric(ytr, model.predict(Xtr));
      point.cv_score += metric(yte, model.predict(Xte));
    }
    const double nf = static_cast<double>(folds.size());
    point.n_train = static_cast<size_t>(std::llround(point.n_train / nf));
    point.train_score /= nf;
    point.cv_score /= nf;
    out.push_back(point);
  }
  return out;
}

} // namespace ml

#endif // HELPER_EVAL_DIAGNOSTICS_HPP
