#include "04_time_series/05_arima/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(ARIMA(p,d,q) models the d-th difference of a series as a linear function of its own p most recent values (autoregression) and the q most recent forecast errors (moving average). Differencing removes trend so that the remaining series has a stable mean and variance; the AR part carries persistence forward; the MA part corrects for recent surprises. Forecasts of the difference are integrated back by adding the last observed level.)LESSON"
      << "\n";
  forecast_course::Arima m(1, 1, 0);
  m.fit({10, 12, 15, 17, 20, 22, 25, 27, 30, 32});
  std::cout << "Constant " << m.constant << ", phi1 " << m.phi[0]
            << ", next forecast " << m.next() << "\n";
}
