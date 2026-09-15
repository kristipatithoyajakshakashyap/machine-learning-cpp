#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(For origin o, train on [0,o) and validate [o,o+12). RMSE=sqrt(mean((y-yhat)^2)). Average the validation-window RMSE to choose regularization. One-step evaluation reveals y_t only after predicting it; multi-step recursive evaluation must instead feed previous forecasts.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
