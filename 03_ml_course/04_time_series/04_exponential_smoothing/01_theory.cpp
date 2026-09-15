#include "04_time_series/04_exponential_smoothing/Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Exponential smoothing forecasts with weighted averages whose weights decay geometrically into the past. Holt-Winters keeps three smoothed components: a level, a trend (slope) and one seasonal offset per season position. The additive form assumes the seasonal swing has constant size; the multiplicative form assumes it scales with the level. Every observation moves each component by a fraction alpha, beta or gamma of its one-step error.)LESSON"
      << "\n";
  forecast_course::HoltWinters m(0.5, 0.1, 0.3, 4);
  m.fit({10, 20, 30, 20, 12, 22, 32, 22});
  std::cout << "Level " << m.level << ", trend " << m.trend
            << ", next forecast " << m.next() << "\n";
}
