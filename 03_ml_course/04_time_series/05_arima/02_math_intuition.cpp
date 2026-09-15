// 03_ml_course/04_time_series/05_arima/02_math_intuition.cpp
// Purpose: ARIMA(1,1,0) worked by hand on eight numbers: difference once,
//          regress each difference on the previous one (AR(1) with constant),
//          forecast the next difference and add the last level back.
// Inputs:  none (series 5,8,12,15,19,22,26,29 is hard-coded).
// Outputs: prints only (differences, c and phi, the hand forecast versus
//          Arima::next(), and the stored residuals).
// Run target: forecast_arima_math_intuition.
#include "04_time_series/05_arima/Model.hpp"
#include <iomanip>
#include <iostream>
// Hand example: difference a short series, fit AR(1) on the differences by
// least squares, and integrate the forecast back.
int main() {
  std::cout
      << R"LESSON(Series 5, 8, 12, 15, 19, 22, 26, 29. First differences w = 3,4,3,4,3,4,3 have no trend. AR(1) with constant regresses w_t on w_{t-1}: the alternating pattern gives a negative phi. The forecast of the next difference is c + phi * w_last; adding the last level 29 integrates it back to the original scale.)LESSON"
      << "\n";
  const ml::Vec y{5, 8, 12, 15, 19, 22, 26, 29};
  // Step 1 - differencing (d = 1) removes the linear growth.
  const ml::Vec w = forecast_course::Arima::difference(y, 1);
  std::cout << "Differences:";
  for (double v : w)
    std::cout << ' ' << v;
  std::cout << "\n";
  // Step 2 - AR(1) on the differences: w_t = c + phi * w_{t-1} + e_t.
  forecast_course::Arima m(1, 1, 0);
  m.fit(y);
  std::cout << std::fixed << std::setprecision(4) << "c = " << m.constant
            << ", phi = " << m.phi[0] << "\n";
  // Step 3 - forecast the next difference by hand and integrate: adding the
  // last observed level undoes the differencing. Must equal m.next().
  const double next_difference = m.constant + m.phi[0] * w.back();
  std::cout << "Next difference " << next_difference << " + last level "
            << y.back() << " = " << next_difference + y.back()
            << " (model says " << m.next() << ")\n";
  // With q = 0 the residuals are not used for forecasting, but they are kept
  // so an MA term could be added later and so save/load is exact.
  std::cout << "Residuals stored for the MA part:";
  for (double e : m.residuals)
    std::cout << ' ' << e;
  std::cout << "\n";
}
