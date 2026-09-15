#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(With a monthly period of 12, yhat[t]=y[t-12]. If January last year had 112 thousand passengers, seasonal naive predicts 112 this January. MASE divides model MAE by in-training seasonal-naive MAE.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
