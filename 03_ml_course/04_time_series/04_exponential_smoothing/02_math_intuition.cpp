// 03_ml_course/04_time_series/04_exponential_smoothing/02_math_intuition.cpp
// Purpose: the additive Holt-Winters recursions traced by hand on eight points
//          with season length four, then compared with what fit() produces.
// Inputs:  none (series 10,20,30,20,12,22,32,22; alpha 0.5, beta 0.1,
//          gamma 0.3 are hard-coded).
// Outputs: prints only (a per-step table of forecast, level, trend, season).
// Run target: forecast_hw_math_intuition.
#include "04_time_series/04_exponential_smoothing/Model.hpp"
#include <iomanip>
#include <iostream>
// Hand example: eight points with season length four, traced step by step.
int main() {
  std::cout
      << R"LESSON(Series 10,20,30,20,12,22,32,22 with season 4. First-season mean 20, second-season mean 22, so level0=20 and trend0=(22-20)/4=0.5. Seasonal offsets average both seasons: (-10,0,10,0). Each step: level=alpha*(y-season)+(1-alpha)*(level+trend); trend=beta*(level-old level)+(1-beta)*trend; season=gamma*(y-level)+(1-gamma)*season.)LESSON"
      << "\n";
  forecast_course::HoltWinters m(0.5, 0.1, 0.3, 4);
  const ml::Vec y{10, 20, 30, 20, 12, 22, 32, 22};
  // Reproduce the initialisation, then trace the recursions one at a time.
  // `m` uses fit() (initialise + observe every point); `trace` starts from
  // the hand-computed initial state and calls observe() explicitly.
  m.fit(ml::Vec(y.begin(), y.begin() + 8));
  forecast_course::HoltWinters trace(0.5, 0.1, 0.3, 4);
  trace.level = 20;
  trace.trend = 0.5;
  trace.seasonal = {-10, 0, 10, 0};
  std::cout << std::fixed << std::setprecision(3)
            << "t  y   forecast  level   trend   season\n";
  for (size_t t = 0; t < y.size(); ++t) {
    const double forecast = trace.next();  // forecast before seeing y[t]
    trace.observe(y[t]);
    std::cout << t << "  " << y[t] << "  " << forecast << "  " << trace.level
              << "  " << trace.trend << "  " << trace.seasonal[t % 4] << "\n";
  }
  // Both objects must end in the same state.
  std::cout << "fit() reproduces the trace: level " << m.level << " vs "
            << trace.level << ", next " << m.next() << " vs " << trace.next()
            << "\n";
}
