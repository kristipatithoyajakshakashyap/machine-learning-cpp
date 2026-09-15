#ifndef HELPER_EVAL_SEARCH_HPP
#define HELPER_EVAL_SEARCH_HPP

// ===========================================================================
// helper/eval/search.hpp  (header-only)
// ---------------------------------------------------------------------------
// Hyper-parameter search over a named grid, evaluated by k-fold CV:
//   grid_search    - sklearn.model_selection.GridSearchCV (exhaustive)
//   random_search  - sklearn.model_selection.RandomizedSearchCV, but sampling
//                    grid combinations WITHOUT replacement
//   nested_cv      - outer CV around an inner random search; reports the
//                    honest (nested) score next to the optimistic naive score
// Every routine is deterministic: folds and samples come from the seed and
// the shared course RNG is re-seeded before every fit.
// The factory receives a Params list (name, value) in grid order and returns
// a model with fit(X, y) and Vec predict(X) const.  metric(y_true, y_pred)
// is higher-is-better.
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/eval/cross_validation.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

namespace ml {

using Params = std::vector<std::pair<std::string, double>>;

struct ParamSpace {
  std::vector<std::pair<std::string, std::vector<double>>> grid;

  // Number of combinations in the full Cartesian product.
  size_t size() const {
    if (grid.empty())
      return 0;
    size_t n = 1;
    for (const auto &axis : grid)
      n *= axis.second.size();
    return n;
  }
  // Decode a linear index (first axis varies fastest) into named values.
  Params at(size_t index) const {
    if (index >= size())
      throw std::out_of_range("ParamSpace index out of range");
    Params out;
    for (const auto &axis : grid) {
      const size_t m = axis.second.size();
      out.emplace_back(axis.first, axis.second[index % m]);
      index /= m;
    }
    return out;
  }
};

struct SearchRow {
  Params params;
  double mean = 0.0; // mean CV metric over the k folds
  double std = 0.0;  // population std over the k folds
};

struct NestedFoldResult {
  size_t fold = 0;
  Params params;            // parameters the inner search selected
  double inner_score = 0.0; // inner-CV mean of the selected parameters
  double outer_score = 0.0; // metric on the untouched outer test fold
};

struct NestedResult {
  std::vector<NestedFoldResult> folds;
  double nested_mean = 0.0; // mean outer score: the honest estimate
  double nested_std = 0.0;
  Params naive_params;      // best of a plain search on ALL rows
  double naive_score = 0.0; // its CV score: optimistic, selection-biased
};

namespace detail {
inline void validate_search_inputs(const Mat &X, const Vec &y,
                                   const ParamSpace &space, size_t k) {
  if (X.empty() || X.size() != y.size())
    throw std::invalid_argument("search needs rows and matching labels");
  if (space.size() == 0)
    throw std::invalid_argument("search needs a non-empty parameter space");
  if (k < 2)
    throw std::invalid_argument("search needs k >= 2 folds");
}

inline std::vector<Fold> make_folds(const Vec &y, size_t k, unsigned seed,
                                    bool classification) {
  return classification ? stratified_kfold(y, k, seed)
                        : kfold(y.size(), k, seed);
}

inline void gather(const Mat &X, const Vec &y, const std::vector<size_t> &rows,
                   Mat &Xo, Vec &yo) {
  Xo.clear();
  yo.clear();
  Xo.reserve(rows.size());
  yo.reserve(rows.size());
  for (size_t i : rows) {
    Xo.push_back(X[i]);
    yo.push_back(y[i]);
  }
}

template <class Factory, class Metric>
SearchRow cv_score(Factory &make, const Mat &X, const Vec &y,
                   const std::vector<Fold> &folds, const Params &params,
                   unsigned seed, Metric metric) {
  Vec scores;
  scores.reserve(folds.size());
  Mat Xtr, Xte;
  Vec ytr, yte;
  for (const auto &fold : folds) {
    gather(X, y, fold.train, Xtr, ytr);
    gather(X, y, fold.test, Xte, yte);
    seed_rng(seed);
    auto model = make(params);
    model.fit(Xtr, ytr);
    const double s = metric(yte, model.predict(Xte));
    if (!std::isfinite(s))
      throw std::runtime_error("search: non-finite fold score");
    scores.push_back(s);
  }
  SearchRow row;
  row.params = params;
  row.mean = mean(scores);
  row.std = std::sqrt(variance(scores, true));
  return row;
}

// First n_iter indices of a seeded shuffle of [0, total): distinct by
// construction.  n_iter >= total degenerates to the exhaustive grid.
inline std::vector<size_t> sample_indices(size_t total, size_t n_iter,
                                          unsigned seed) {
  std::vector<size_t> ids(total);
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(std::min(n_iter, total));
  return ids;
}

template <class Factory, class Metric>
std::vector<SearchRow>
evaluate_indices(Factory make, const Mat &X, const Vec &y,
                 const ParamSpace &space, const std::vector<size_t> &ids,
                 size_t k, unsigned seed, Metric metric, bool classification) {
  const auto folds = make_folds(y, k, seed, classification);
  std::vector<SearchRow> out;
  out.reserve(ids.size());
  for (size_t id : ids)
    out.push_back(cv_score(make, X, y, folds, space.at(id), seed, metric));
  return out;
}
} // namespace detail

inline const SearchRow &best_row(const std::vector<SearchRow> &rows) {
  if (rows.empty())
    throw std::invalid_argument("best_row needs at least one row");
  return *std::max_element(
      rows.begin(), rows.end(),
      [](const SearchRow &a, const SearchRow &b) { return a.mean < b.mean; });
}

template <class Factory, class Metric>
std::vector<SearchRow> grid_search(Factory make, const Mat &X, const Vec &y,
                                   const ParamSpace &space, size_t k,
                                   unsigned seed, Metric metric,
                                   bool classification) {
  detail::validate_search_inputs(X, y, space, k);
  std::vector<size_t> ids(space.size());
  std::iota(ids.begin(), ids.end(), 0);
  return detail::evaluate_indices(make, X, y, space, ids, k, seed, metric,
                                  classification);
}

template <class Factory, class Metric>
std::vector<SearchRow> random_search(Factory make, const Mat &X, const Vec &y,
                                     const ParamSpace &space, size_t n_iter,
                                     size_t k, unsigned seed, Metric metric,
                                     bool classification) {
  detail::validate_search_inputs(X, y, space, k);
  if (n_iter == 0)
    throw std::invalid_argument("random_search needs n_iter >= 1");
  const auto ids = detail::sample_indices(space.size(), n_iter, seed);
  return detail::evaluate_indices(make, X, y, space, ids, k, seed, metric,
                                  classification);
}

template <class Factory, class Metric>
NestedResult nested_cv(Factory make, const Mat &X, const Vec &y,
                       const ParamSpace &space, size_t outer_k, size_t inner_k,
                       size_t n_iter, unsigned seed, Metric metric,
                       bool classification) {
  detail::validate_search_inputs(X, y, space, outer_k);
  if (inner_k < 2 || n_iter == 0)
    throw std::invalid_argument(
        "nested_cv needs inner_k >= 2 and n_iter >= 1");
  const auto outer = detail::make_folds(y, outer_k, seed, classification);
  NestedResult result;
  Vec outer_scores;
  Mat Xtr, Xte;
  Vec ytr, yte;
  for (size_t f = 0; f < outer.size(); ++f) {
    detail::gather(X, y, outer[f].train, Xtr, ytr);
    detail::gather(X, y, outer[f].test, Xte, yte);
    const auto rows =
        random_search(make, Xtr, ytr, space, n_iter, inner_k,
                      seed + static_cast<unsigned>(f), metric, classification);
    const SearchRow &best = best_row(rows);
    seed_rng(seed);
    auto model = make(best.params);
    model.fit(Xtr, ytr);
    NestedFoldResult r;
    r.fold = f;
    r.params = best.params;
    r.inner_score = best.mean;
    r.outer_score = metric(yte, model.predict(Xte));
    outer_scores.push_back(r.outer_score);
    result.folds.push_back(r);
  }
  result.nested_mean = mean(outer_scores);
  result.nested_std = std::sqrt(variance(outer_scores, true));
  const auto naive = random_search(make, X, y, space, n_iter, inner_k, seed,
                                   metric, classification);
  const SearchRow &nb = best_row(naive);
  result.naive_params = nb.params;
  result.naive_score = nb.mean;
  return result;
}

inline std::string params_to_string(const Params &p, char sep = ';') {
  std::string out;
  for (size_t i = 0; i < p.size(); ++i) {
    if (i)
      out += sep;
    out += p[i].first + '=' + std::to_string(p[i].second);
  }
  return out;
}

inline double param_value(const Params &p, const std::string &name) {
  for (const auto &kv : p)
    if (kv.first == name)
      return kv.second;
  throw std::invalid_argument("unknown parameter: " + name);
}

} // namespace ml

#endif // HELPER_EVAL_SEARCH_HPP
