// 03_ml_course/04_time_series/05_arima/tests/model_test.cpp
// Purpose: numerical unit test for the ARIMA model: least-squares recovery
//          of a known AR(1) coefficient, the differencing/integration round
//          trip on a quadratic series, and exact save/load.
// Inputs:  none (a seeded AR(1) sample and a quadratic series are built in
//          code; no files are read).
// Outputs: prints only (nothing on success; throws on the first failed check,
//          which makes the process exit non-zero so ctest reports it).
// Run target: forecast_arima_tests  (ctest -R forecast_arima_numerical).
#include "04_time_series/05_arima/Model.hpp"
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
int main() {
  // 1. AR(1) with phi = 0.7 and seeded Gaussian noise: OLS must recover phi.
  //    500 points keep the sampling error of the estimate well under 0.1.
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 1.0);
  ml::Vec x{0};
  for (size_t t = 1; t < 500; ++t)
    x.push_back(0.7 * x.back() + noise(rng));
  forecast_course::Arima ar(1, 0, 0);
  ar.fit(x);
  if (std::abs(ar.phi[0] - 0.7) > 0.1)
    throw std::runtime_error("AR(1) coefficient not recovered");
  // With q = 0 no MA coefficients may be allocated.
  if (ar.theta.size() != 0)
    throw std::runtime_error("q=0 must have no MA coefficients");
  // 2. Differencing and integration round trip. Second differences of
  //    t^2 are the constant 2, and the 10-point series loses 2 elements.
  const ml::Vec quadratic{1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
  const ml::Vec second = forecast_course::Arima::difference(quadratic, 2);
  if (second.size() != 8)
    throw std::runtime_error("Second difference length");
  for (double v : second)
    if (std::abs(v - 2) > 1e-12)
      throw std::runtime_error("Second difference of squares must be 2");
  // ARIMA(0,2,0) forecasts the mean second difference (2) and integrates it
  // twice: 100 + (100 - 81) + 2 = 121 = 11^2.
  forecast_course::Arima walk(0, 2, 0);
  walk.fit(quadratic);
  if (std::abs(walk.next() - 121) > 1e-9)
    throw std::runtime_error("Integration of the forecast difference");
  // Observing exactly the forecast must give a zero residual and shift the
  // integration state so the next forecast is 12^2 = 144.
  walk.observe(121);
  if (std::abs(walk.next() - 144) > 1e-9 || std::abs(walk.residuals.back()) > 1e-9)
    throw std::runtime_error("observe() must integrate consistently");
  // 3. Save/load must be exact: orders, MA coefficients and the residual
  //    history (needed by the MA part) all round-trip through text.
  forecast_course::Arima full(2, 1, 1);
  full.fit(x);
  std::stringstream s;
  full.save(s);
  forecast_course::Arima copy;
  copy.load(s);
  if (copy.next() != full.next() || copy.p != 2 || copy.d != 1 || copy.q != 1 ||
      copy.theta != full.theta || copy.residuals != full.residuals)
    throw std::runtime_error("Reload");
}
