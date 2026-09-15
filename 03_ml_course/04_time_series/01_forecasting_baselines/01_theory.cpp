#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Forecasts must beat simple references. Mean forecasts ignore dynamics; last-value forecasts extrapolate the most recent observation; seasonal naive repeats the previous seasonal position. Use only information observed at the forecast origin.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
