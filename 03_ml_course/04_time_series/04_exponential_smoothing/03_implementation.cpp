// 03_ml_course/04_time_series/04_exponential_smoothing/03_implementation.cpp
// Purpose: lesson on additive Holt-Winters for AirPassengers. One model is
//          fitted normally; a second copy replays the level/trend/season
//          recursions one observation at a time so every component can be
//          recorded and plotted alongside the one-step-ahead forecast.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly totals, 1949-1960);
//          smoothing weights alpha 0.5, beta 0.1, gamma 0.3, period 12.
// Outputs: results/03_implementation_results/{components.csv,
//          figures/level.svg, figures/trend.svg, figures/season.svg};
//          prints the final level, trend, seasonal offsets and next forecast.
// Run target: forecast_hw_implementation.
#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
#include <iostream>
#include <sstream>
// Fit additive Holt-Winters on AirPassengers and expose the components.
int main() {
  try {
    const auto y = forecast_course::read_passengers(
        std::string(DATA_DIR) + "/air_passengers.csv");
    // Reference fit: fit() initialises from the first two seasons and then
    // runs the recursions over the whole series.
    forecast_course::HoltWinters model(0.5, 0.1, 0.3, 12);
    model.fit(y);
    // Replay the recursions from the two-season initial state to record the
    // path of every component.
    forecast_course::HoltWinters replay(0.5, 0.1, 0.3, 12);
    replay.initialize(y);
    ml::Vec time, level, trend, season, fitted;
    std::ostringstream csv;
    csv << "month_index,actual,one_step_forecast,level,trend,season\n";
    for (size_t t = 0; t < y.size(); ++t) {
      // next() is the forecast made BEFORE seeing y[t]; observe() then
      // updates level, trend and the seasonal slot t % 12.
      const double forecast = replay.next();
      replay.observe(y[t]);
      time.push_back(double(t));
      level.push_back(replay.level);
      trend.push_back(replay.trend);
      season.push_back(replay.seasonal[t % 12]);
      fitted.push_back(forecast);
      csv << t << ',' << y[t] << ',' << forecast << ',' << replay.level << ','
          << replay.trend << ',' << replay.seasonal[t % 12] << '\n';
    }
    // RUN_OUTPUT_DIR is injected by CMake (results/03_implementation_results).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("components.csv", csv.str());
    // Three figures: level vs actual vs one-step forecast, the trend slope,
    // and the seasonal offset that was active at each month.
    ml::Plot chart;
    chart.line(time, y, "actual");
    chart.line(time, level, "level");
    chart.line(time, fitted, "one-step forecast");
    chart.title("Holt-Winters level against observations");
    a.figure("level.svg", chart);
    ml::Plot slope;
    slope.line(time, trend, "trend per month");
    a.figure("trend.svg", slope);
    ml::Plot seasonal;
    seasonal.line(time, season, "seasonal offset");
    a.figure("season.svg", seasonal);
    // The replayed model must end in the same state as the reference fit,
    // so both next() values printed here should agree.
    std::cout << "Final level " << model.level << ", trend " << model.trend
              << " per month\nSeasonal offsets:";
    for (double s : model.seasonal)
      std::cout << ' ' << s;
    std::cout << "\nNext forecast " << model.next()
              << " (replayed " << replay.next() << ")\n";
    return 0;
  } catch (const std::exception &e) {
    // Any I/O or validation error is reported once and mapped to exit code 1.
    std::cerr << e.what() << '\n';
    return 1;
  }
}
