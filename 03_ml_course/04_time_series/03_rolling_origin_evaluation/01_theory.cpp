#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Random splits scramble the forecasting problem and can train on future regimes. Expanding-window validation repeatedly fits on a prefix and evaluates the next time block. Final chronological holdout remains untouched during parameter selection.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
