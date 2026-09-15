// 03_ml_course/04_time_series/01_forecasting_baselines/03_implementation.cpp
// Purpose: lesson on forecasting baselines for AirPassengers. After a
//          three-point sanity check, the last 20% of the series is forecast
//          one step ahead by three baselines - naive (last value), seasonal
//          naive (same month last year) and the expanding mean - and their
//          RMSE/MAE are tabulated. Any real model must beat these.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly totals, 1949-1960).
// Outputs: results/03_implementation_results/{forecasts.csv, rmse_table.csv,
//          figures/baselines.svg}; also prints the three RMSEs.
// Run target: forecast_baselines_implementation.
#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
int main() {
  std::cout
      << R"LESSON(Predict before observing the new target; then append the target to the historical buffer.)LESSON"
      << "\n";
  // Sanity check: season 3 on {10, 20, 30} must forecast 10 (three steps back).
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";

  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  const ml::Vec y = forecast_course::read_passengers(
      std::string(DATA_DIR) + "/air_passengers.csv");
  // Chronological 80/20 split; the holdout is never shuffled.
  const size_t cut = y.size() * 4 / 5;
  const ml::Vec training(y.begin(), y.begin() + cut);
  // period 1 = last value; period 12 = same month one year earlier.
  forecast_course::SeasonalNaive naive(1), seasonal(12);
  naive.fit(training);
  seasonal.fit(training);
  // Expanding mean: running sum of everything observed so far.
  double running_sum = 0;
  for (double v : training)
    running_sum += v;
  ml::Vec t, actual, f_naive, f_seasonal, f_mean;
  std::ostringstream rows;
  rows << std::setprecision(17)
       << "month_index,actual,naive,seasonal_naive,mean\n";
  // One-step protocol: forecast month i, record it, then reveal y[i].
  for (size_t i = cut; i < y.size(); ++i) {
    const double mean_forecast = running_sum / double(i);
    t.push_back(double(i));
    actual.push_back(y[i]);
    f_naive.push_back(naive.next());
    f_seasonal.push_back(seasonal.next());
    f_mean.push_back(mean_forecast);
    rows << i << ',' << y[i] << ',' << naive.next() << ',' << seasonal.next()
         << ',' << mean_forecast << '\n';
    naive.observe(y[i]);
    seasonal.observe(y[i]);
    running_sum += y[i];
  }
  a.write("forecasts.csv", rows.str());
  // Error table: seasonal naive should win clearly on this seasonal series.
  std::ostringstream table;
  table << std::setprecision(17) << "baseline,rmse,mae\n"
        << "naive," << ml::rmse(actual, f_naive) << ',' << ml::mae(actual, f_naive)
        << "\nseasonal_naive," << ml::rmse(actual, f_seasonal) << ','
        << ml::mae(actual, f_seasonal) << "\nmean," << ml::rmse(actual, f_mean)
        << ',' << ml::mae(actual, f_mean) << '\n';
  a.write("rmse_table.csv", table.str());
  std::cout << std::setprecision(5) << "held-out months " << y.size() - cut
            << "\n  naive RMSE          " << ml::rmse(actual, f_naive)
            << "\n  seasonal naive RMSE " << ml::rmse(actual, f_seasonal)
            << "\n  mean RMSE           " << ml::rmse(actual, f_mean) << '\n';
  ml::Plot chart;
  chart.title("AirPassengers holdout: baselines versus actual");
  chart.xlabel("month index");
  chart.ylabel("passengers (thousands)");
  chart.line(t, actual, "actual");
  chart.line(t, f_naive, "naive");
  chart.line(t, f_seasonal, "seasonal naive");
  chart.line(t, f_mean, "mean");
  a.figure("figures/baselines.svg", chart);
  std::cout << "Saved forecasts.csv, rmse_table.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
