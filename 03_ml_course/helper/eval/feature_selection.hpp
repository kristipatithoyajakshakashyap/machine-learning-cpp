#ifndef HELPER_EVAL_FEATURE_SELECTION_HPP
#define HELPER_EVAL_FEATURE_SELECTION_HPP

// ===========================================================================
// helper/eval/feature_selection.hpp  (header-only)
// ---------------------------------------------------------------------------
// Three families of feature selection:
//   filter_scores      - ANOVA F (classification) or |Pearson r| (regression)
//                        per column; no model involved.
//   forward_selection  - greedy wrapper: add the feature that most improves
//                        the cross-validated metric of a model factory.
//   l1_path            - embedded: lasso coefficients over a lambda grid on
//                        standardized columns (reuses the course lasso in
//                        01_regression/01_linear_regression).
// Deterministic: folds come from the caller's seed and the shared course RNG
// is re-seeded before every fit so stochastic models replay exactly.
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "01_supervised/01_regression/01_linear_regression/LinearRegression.hpp"
#include "helper/eval/cross_validation.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

namespace ml {

namespace detail {
inline void validate_selection_data(const Mat &X, const Vec &y) {
  if (X.empty() || X[0].empty() || X.size() != y.size())
    throw std::invalid_argument("feature selection needs data and labels");
  for (const auto &row : X) {
    if (row.size() != X[0].size())
      throw std::invalid_argument("feature selection: ragged matrix");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("feature selection: nonfinite feature");
  }
  for (double v : y)
    if (!std::isfinite(v))
      throw std::invalid_argument("feature selection: nonfinite target");
}
inline Mat project(const Mat &X, const std::vector<size_t> &cols) {
  Mat out(X.size(), Vec(cols.size()));
  for (size_t i = 0; i < X.size(); ++i)
    for (size_t j = 0; j < cols.size(); ++j)
      out[i][j] = X[i][cols[j]];
  return out;
}
inline Vec column(const Mat &X, size_t j) {
  Vec out(X.size());
  for (size_t i = 0; i < X.size(); ++i)
    out[i] = X[i][j];
  return out;
}
// One-way ANOVA F statistic of column values grouped by class label.
inline double anova_f(const Vec &x, const Vec &y) {
  std::map<double, std::pair<double, size_t>> groups; // sum, count
  for (size_t i = 0; i < x.size(); ++i) {
    groups[y[i]].first += x[i];
    groups[y[i]].second += 1;
  }
  const size_t k = groups.size(), n = x.size();
  if (k < 2 || n <= k)
    return 0.0;
  const double grand = mean(x);
  double between = 0.0, within = 0.0;
  for (const auto &kv : groups) {
    const double gm = kv.second.first / static_cast<double>(kv.second.second);
    between += static_cast<double>(kv.second.second) * (gm - grand) * (gm - grand);
  }
  for (size_t i = 0; i < n; ++i) {
    const auto &g = groups[y[i]];
    const double gm = g.first / static_cast<double>(g.second);
    within += (x[i] - gm) * (x[i] - gm);
  }
  const double df_b = static_cast<double>(k - 1), df_w = static_cast<double>(n - k);
  if (within <= 0.0)
    return between > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  return (between / df_b) / (within / df_w);
}
inline double abs_pearson(const Vec &x, const Vec &y) {
  const double mx = mean(x), my = mean(y);
  double sxy = 0.0, sxx = 0.0, syy = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  if (sxx <= 0.0 || syy <= 0.0)
    return 0.0;
  return std::fabs(sxy / std::sqrt(sxx * syy));
}
} // namespace detail

// Univariate relevance of every column: ANOVA F for classification, |r| for
// regression. Higher is more relevant; a constant column scores 0.
inline Vec filter_scores(const Mat &X, const Vec &y, bool classification) {
  detail::validate_selection_data(X, y);
  Vec scores(X[0].size());
  for (size_t j = 0; j < scores.size(); ++j) {
    const Vec x = detail::column(X, j);
    scores[j] = classification ? detail::anova_f(x, y) : detail::abs_pearson(x, y);
  }
  return scores;
}

// Indices of the k largest scores, ties broken by lower index.
inline std::vector<size_t> top_k(const Vec &scores, size_t k) {
  if (k == 0 || k > scores.size())
    throw std::invalid_argument("top_k needs 1 <= k <= number of features");
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return scores[a] > scores[b]; });
  order.resize(k);
  return order;
}

struct ForwardStep {
  size_t feature = 0;  // column added at this step
  double cv_score = 0; // CV metric with all features chosen so far
};

// Greedy forward wrapper. make() -> model with fit(X, y) and predict(X);
// metric(y_true, y_pred) is higher-is-better. Returns k_max steps in the
// order the features were added.
template <class Factory, class Metric>
std::vector<ForwardStep>
forward_selection(Factory make, const Mat &X, const Vec &y, size_t k_max,
                  size_t folds, unsigned seed, Metric metric,
                  bool stratified = true) {
  detail::validate_selection_data(X, y);
  const size_t p = X[0].size();
  if (k_max == 0 || k_max > p || folds < 2)
    throw std::invalid_argument(
        "forward_selection needs 1 <= k_max <= features and folds >= 2");
  const auto split = stratified ? stratified_kfold(y, folds, seed)
                                : kfold(X.size(), folds, seed);
  auto cv_score = [&](const std::vector<size_t> &cols) {
    const Mat Xs = detail::project(X, cols);
    double total = 0.0;
    for (const auto &fold : split) {
      Mat Xtr, Xte;
      Vec ytr, yte;
      for (size_t i : fold.train) {
        Xtr.push_back(Xs[i]);
        ytr.push_back(y[i]);
      }
      for (size_t i : fold.test) {
        Xte.push_back(Xs[i]);
        yte.push_back(y[i]);
      }
      seed_rng(seed);
      auto model = make();
      model.fit(Xtr, ytr);
      total += metric(yte, model.predict(Xte));
    }
    return total / static_cast<double>(split.size());
  };
  std::vector<size_t> chosen;
  std::vector<bool> used(p, false);
  std::vector<ForwardStep> out;
  for (size_t step = 0; step < k_max; ++step) {
    ForwardStep best;
    best.cv_score = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < p; ++j) {
      if (used[j])
        continue;
      std::vector<size_t> trial = chosen;
      trial.push_back(j);
      const double s = cv_score(trial);
      if (s > best.cv_score) {
        best.cv_score = s;
        best.feature = j;
      }
    }
    used[best.feature] = true;
    chosen.push_back(best.feature);
    out.push_back(best);
  }
  return out;
}

// Lasso coefficient path: out[l][j] is the slope of standardized column j at
// lambdas[l]. Features that stay non-zero at large lambda are the most
// relevant. For classification pass 0/1 labels (linear-probability lasso).
inline Mat l1_path(const Mat &X, const Vec &y, const Vec &lambdas) {
  detail::validate_selection_data(X, y);
  if (lambdas.empty())
    throw std::invalid_argument("l1_path needs at least one lambda");
  const Mat Z = standardize(X);
  Mat out;
  for (double lambda : lambdas) {
    if (!std::isfinite(lambda) || lambda < 0)
      throw std::invalid_argument("l1_path lambdas must be non-negative");
    LinearRegression lasso;
    lasso.fit_lasso(Z, y, lambda);
    const Vec &w = lasso.coef();
    out.emplace_back(w.begin() + 1, w.end());
  }
  return out;
}

// Rank features by the largest lambda at which they are still non-zero on the
// path (ties broken by |coef| at the smallest lambda). Returns one rank score
// per feature, higher is more relevant.
inline Vec l1_scores(const Mat &path, const Vec &lambdas) {
  if (path.empty() || path.size() != lambdas.size())
    throw std::invalid_argument("l1_scores needs one path row per lambda");
  const size_t p = path[0].size();
  Vec scores(p, 0.0);
  size_t smallest = 0;
  for (size_t l = 1; l < lambdas.size(); ++l)
    if (lambdas[l] < lambdas[smallest])
      smallest = l;
  for (size_t j = 0; j < p; ++j) {
    double entry = 0.0;
    for (size_t l = 0; l < lambdas.size(); ++l)
      if (std::fabs(path[l][j]) > 1e-12)
        entry = std::max(entry, lambdas[l]);
    scores[j] = entry + 1e-6 * std::fabs(path[smallest][j]);
  }
  return scores;
}

} // namespace ml

#endif // HELPER_EVAL_FEATURE_SELECTION_HPP
