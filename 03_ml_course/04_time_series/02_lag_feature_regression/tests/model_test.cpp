// 03_ml_course/04_time_series/02_lag_feature_regression/tests/model_test.cpp
// Purpose: numerical unit test shared by the first three time-series modules.
//          It checks the SeasonalNaive baseline, the lag-feature builder and
//          LagRegression (fit, forecast, save/load) on tiny hand-made series.
// Inputs:  none (all series are built in code; no files are read).
// Outputs: prints only (nothing on success; throws with a message on failure,
//          which makes the process exit non-zero so ctest reports the failure).
// Run target: forecast_lag_tests  (ctest -R forecast_lag_numerical).
#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
#include <stdexcept>
// Each check throws std::runtime_error naming the property that failed.
int main() {
  // SeasonalNaive(3): the forecast is the value one season (3 steps) back.
  // After fitting {1,2,3} the next point (index 3) copies index 0 -> 1.
  forecast_course::SeasonalNaive b(3);
  b.fit({1, 2, 3});
  if (b.next() != 1)
    throw std::runtime_error("Seasonal lookup");
  // observe() appends to the history, so index 4 copies index 1 -> 2.
  b.observe(4);
  if (b.next() != 2)
    throw std::runtime_error("Observed history");
  // A pure linear series y_t = 2t + 10 with 48 points (four "years").
  ml::Vec y;
  for (int i = 0; i < 48; ++i)
    y.push_back(2 * i + 10);
  // features(y, t) builds the lag vector for predicting y[t]: feature 0 is
  // lag 1 (y[t-1]) and feature 2 is lag 12 (y[t-12]); see Model.hpp.
  auto x = forecast_course::features(y, 24);
  if (x[0] != y[23] || x[2] != y[12])
    throw std::runtime_error("Lag indexing");
  // Unregularised (alpha = 0) lag regression must extrapolate the trend:
  // the next value is 2 * 48 + 10 = 106 (tolerance 0.1 for round-off).
  forecast_course::LagRegression m(0);
  m.fit(y);
  if (std::abs(m.next() - 106) > 0.1)
    throw std::runtime_error("Trend forecast");
  // Text save/load round trip: the reloaded model must forecast identically.
  std::stringstream s;
  m.save(s);
  forecast_course::LagRegression c;
  c.load(s);
  if (std::abs(c.next() - m.next()) > 1e-12)
    throw std::runtime_error("Reload");
}
