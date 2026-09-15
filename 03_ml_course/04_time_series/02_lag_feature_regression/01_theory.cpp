#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Convert an ordered series to supervised rows using lagged targets, time and seasonal features. A row for month t may use t-1, t-2 and t-12, never the target at t or later. Fit imputation/scaling and ridge regression on historical rows only.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
