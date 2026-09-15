#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(x_t=[y_(t-1),y_(t-2),y_(t-12),t,sin(2*pi*t/12),cos(2*pi*t/12)]. Fit squared loss plus an L2 penalty. Sine and cosine avoid a discontinuity between December and January; lag-12 captures yearly repetition.)LESSON"
      << "\n";
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";
}
