// 03_ml_course/04_time_series/03_rolling_origin_evaluation/03_implementation.cpp
// Purpose: lesson on rolling-origin evaluation. After the baseline sanity
//          check each candidate forecaster (seasonal naive and the three-value
//          lag-regression grid) is refitted at every origin on an expanding
//          window and scored on the next twelve months; per-origin RMSE is
//          what selects a model honestly, without touching the holdout.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly totals, 1949-1960).
// Outputs: results/03_implementation_results/{per_origin_errors.csv,
//          figures/rmse_per_origin.svg}; also prints mean RMSE per candidate.
// Run target: forecast_rolling_implementation.
#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
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

  using namespace forecast_course;
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  const ml::Vec y = read_passengers(std::string(DATA_DIR) + "/air_passengers.csv");
  // Origins only advance up to `cut`, so the final 20% holdout stays unseen.
  const size_t cut = y.size() * 4 / 5, horizon = 12;
  // A candidate is a factory from Workflow.hpp plus its grid index.
  struct Candidate {
    std::string name;
    std::unique_ptr<Forecast> (*make)(size_t);
    size_t index;
  };
  const std::vector<Candidate> candidates = {
      {"seasonal_naive", make_seasonal, 0},
      {"lag_alpha_" + lag_values(0), make_lag, 0},
      {"lag_alpha_" + lag_values(1), make_lag, 1},
      {"lag_alpha_" + lag_values(2), make_lag, 2}};
  std::ostringstream csv;
  csv << std::setprecision(17) << "candidate,origin,validation_end,rmse,mae\n";
  ml::Plot chart;
  chart.title("Rolling-origin RMSE per candidate (12-month windows)");
  chart.xlabel("origin (month index)");
  chart.ylabel("RMSE");
  std::cout << std::setprecision(5) << "mean RMSE over origins\n";
  for (const auto &c : candidates) {
    ml::Vec origins, scores;
    // Expanding window: fit on months [0, origin), forecast the next 12 one
    // step at a time, then move the origin forward by a full year.
    for (size_t origin = 48; origin + horizon <= cut; origin += horizon) {
      auto model = c.make(c.index);
      model->fit(ml::Vec(y.begin(), y.begin() + origin));
      ml::Vec actual, pred;
      for (size_t t = origin; t < origin + horizon; ++t) {
        pred.push_back(model->next());
        actual.push_back(y[t]);
        model->observe(y[t]);
      }
      const double score = ml::rmse(actual, pred);
      csv << c.name << ',' << origin << ',' << origin + horizon << ',' << score
          << ',' << ml::mae(actual, pred) << '\n';
      origins.push_back(double(origin));
      scores.push_back(score);
    }
    chart.line(origins, scores, c.name);
    std::cout << "  " << c.name << " " << ml::mean(scores) << '\n';
  }
  a.write("per_origin_errors.csv", csv.str());
  a.figure("figures/rmse_per_origin.svg", chart);
  std::cout << "Saved per_origin_errors.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
