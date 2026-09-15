// 03_ml_course/04_time_series/02_lag_feature_regression/03_implementation.cpp
// Purpose: lesson on regression with lag features. After the baseline sanity
//          check the AirPassengers series is turned into a supervised table
//          (lag_1, lag_2, lag_12, month index, seasonal sin/cos); a ridge-style
//          LagRegression (alpha 0.1) is fitted on the first 80% and produces
//          one-step forecasts for the rest.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly totals, 1949-1960).
// Outputs: results/03_implementation_results/{lag_features.csv,
//          coefficients.csv, fitted_vs_actual.csv,
//          figures/fitted_vs_actual.svg}; also prints the coefficients and
//          the held-out RMSE/MAE.
// Run target: forecast_lag_implementation.
#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include "04_time_series/02_lag_feature_regression/Model.hpp"
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
  // Sanity check: season 3 on {10, 20, 30} must forecast 10.
  forecast_course::SeasonalNaive m(3);
  m.fit({10, 20, 30});
  std::cout << "Next forecast: " << m.next() << "\n";

  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  const ml::Vec y = forecast_course::read_passengers(
      std::string(DATA_DIR) + "/air_passengers.csv");
  const size_t cut = y.size() * 4 / 5;  // chronological 80/20 split
  const ml::Vec training(y.begin(), y.begin() + cut);
  const std::vector<std::string> names = {"lag_1",       "lag_2",
                                          "lag_12",      "month_index",
                                          "season_sin",  "season_cos"};
  // The supervised view of the whole series: one row per month t >= 12,
  // features built only from months before t, target y[t].
  std::ostringstream table;
  table << std::setprecision(17) << "month_index";
  for (const auto &n : names)
    table << ',' << n;
  table << ",target\n";
  for (size_t t = 12; t < y.size(); ++t) {
    table << t;
    for (double v : forecast_course::features(y, t))
      table << ',' << v;
    table << ',' << y[t] << '\n';
  }
  a.write("lag_features.csv", table.str());

  // Fit on training months only; coefficients are on standardized features,
  // so their magnitudes are comparable across features.
  forecast_course::LagRegression model(0.1);
  model.fit(training);
  std::ostringstream coef;
  coef << std::setprecision(17) << "feature,coefficient\n"
       << "intercept," << model.regression.intercept << '\n';
  std::cout << std::setprecision(5) << "standardised coefficients\n";
  for (size_t j = 0; j < names.size(); ++j) {
    coef << names[j] << ',' << model.regression.coef[j] << '\n';
    std::cout << "  " << names[j] << " " << model.regression.coef[j] << '\n';
  }
  a.write("coefficients.csv", coef.str());

  // In-sample fitted values on the training rows, then one-step forecasts.
  ml::Vec t, actual, fitted, split;
  std::ostringstream rows;
  rows << std::setprecision(17) << "month_index,split,actual,fitted\n";
  for (size_t i = 12; i < cut; ++i) {
    const double f = model.regression.predict(model.preprocessing.transform(
        {forecast_course::features(training, i)}))[0];
    t.push_back(double(i));
    actual.push_back(y[i]);
    fitted.push_back(f);
    rows << i << ",train," << y[i] << ',' << f << '\n';
  }
  // Holdout: forecast, record, then observe() so the lags advance. The
  // coefficients stay frozen throughout.
  ml::Vec test_actual, test_pred;
  for (size_t i = cut; i < y.size(); ++i) {
    const double f = model.next();
    t.push_back(double(i));
    actual.push_back(y[i]);
    fitted.push_back(f);
    test_actual.push_back(y[i]);
    test_pred.push_back(f);
    rows << i << ",test," << y[i] << ',' << f << '\n';
    model.observe(y[i]);
  }
  a.write("fitted_vs_actual.csv", rows.str());
  std::cout << "held-out one-step RMSE " << ml::rmse(test_actual, test_pred)
            << "  MAE " << ml::mae(test_actual, test_pred) << '\n';
  ml::Plot chart;
  chart.title("Lag regression: fitted / forecast versus actual");
  chart.xlabel("month index");
  chart.ylabel("passengers (thousands)");
  chart.line(t, actual, "actual");
  chart.line(t, fitted, "fitted");
  a.figure("figures/fitted_vs_actual.svg", chart);
  std::cout << "Saved lag_features.csv, coefficients.csv, "
               "fitted_vs_actual.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
