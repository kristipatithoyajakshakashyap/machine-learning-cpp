// 03_ml_course/04_time_series/04_exponential_smoothing/tests/model_test.cpp
// Purpose: numerical unit test for additive Holt-Winters: it must beat the
//          seasonal-naive baseline on a synthetic trend+season series, its
//          observe() must match fit(), the level recursion must follow the
//          textbook formula, and save/load must be exact.
// Inputs:  none (a 96-point synthetic monthly series is built in code).
// Outputs: prints only (nothing on success; throws on the first failed check,
//          which makes the process exit non-zero so ctest reports it).
// Run target: forecast_hw_tests  (ctest -R forecast_hw_numerical).
#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include "04_time_series/04_exponential_smoothing/Model.hpp"
#include "helper/math/metrics.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
namespace {
// Synthetic monthly series: linear trend plus a fixed additive season.
// Parameters: n - number of points. Returns y_t = 50 + 0.8 t + pattern[t%12].
ml::Vec synthetic(size_t n) {
  const double pattern[12] = {-8, -6, -2, 1, 4, 9, 12, 11, 5, 0, -5, -9};
  ml::Vec y;
  for (size_t t = 0; t < n; ++t)
    y.push_back(50 + 0.8 * double(t) + pattern[t % 12]);
  return y;
}
// Rolling one-step hold-out: fit on all but the last `last` points, then
// forecast each held-out point and feed the truth back with observe().
// Parameters: model - any forecaster with fit/next/observe (taken by value so
//             the caller's copy is untouched); y - full series; last - hold-out
//             length. Returns the RMSE over the hold-out. Complexity O(n).
template <class M> double holdout_rmse(M model, const ml::Vec &y, size_t last) {
  model.fit(ml::Vec(y.begin(), y.begin() + (y.size() - last)));
  ml::Vec actual, pred;
  for (size_t t = y.size() - last; t < y.size(); ++t) {
    pred.push_back(model.next());
    actual.push_back(y[t]);
    model.observe(y[t]);
  }
  return ml::rmse(actual, pred);
}
} // namespace
int main() {
  const ml::Vec y = synthetic(96);
  // 1. Skill: on a trending seasonal series Holt-Winters (which tracks the
  //    trend) must have lower hold-out RMSE than seasonal naive (which lags
  //    the trend by one full year).
  const double hw = holdout_rmse(forecast_course::HoltWinters(0.5, 0.1, 0.3),
                                 y, 24);
  const double naive = holdout_rmse(forecast_course::SeasonalNaive(12), y, 24);
  if (!(hw < naive))
    throw std::runtime_error("Holt-Winters should beat seasonal naive");
  forecast_course::HoltWinters m(0.5, 0.1, 0.3, 12);
  m.fit(y);
  // 2. observe() must advance the state exactly as fit() would have:
  //    fit(all but last) + observe(last) == fit(all).
  forecast_course::HoltWinters shorter(0.5, 0.1, 0.3, 12);
  shorter.fit(ml::Vec(y.begin(), y.end() - 1));
  shorter.observe(y.back());
  if (std::abs(shorter.next() - m.next()) > 1e-9)
    throw std::runtime_error("observe() inconsistent with fit()");
  // 3. Level recursion: level' = alpha * (y - s_old) + (1 - alpha) * (l + b).
  //    Feed an observation 3 above the forecast and check the new level.
  const double season_old = m.seasonal[m.observed % 12];
  const double expected_level =
      0.5 * (m.next() + 3 - season_old) + 0.5 * (m.level + m.trend);
  m.observe(m.next() + 3);
  if (std::abs(m.level - expected_level) > 1e-9)
    throw std::runtime_error("Level recursion");
  // 4. Persistence: a reloaded model must reproduce next(), the observation
  //    counter and the hyper-parameters bit-for-bit.
  std::stringstream s;
  m.save(s);
  forecast_course::HoltWinters c;
  c.load(s);
  if (c.next() != m.next() || c.observed != m.observed || c.alpha != 0.5)
    throw std::runtime_error("Reload");
}
