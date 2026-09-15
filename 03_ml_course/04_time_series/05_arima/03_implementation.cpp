// 03_ml_course/04_time_series/05_arima/03_implementation.cpp
// Purpose: lesson on ARIMA(1,1,1) for AirPassengers. Shows why differencing
//          is needed (the raw series has slowly decaying autocorrelation, the
//          differenced one does not) and inspects the in-sample residuals.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly totals, 1949-1960);
//          orders p=1, d=1, q=1 and 24 autocorrelation lags are hard-coded.
// Outputs: results/03_implementation_results/{differences.csv,
//          autocorrelation.csv, figures/differences.svg,
//          figures/autocorrelation.svg, figures/residuals.svg};
//          prints the coefficients, next forecast and lag-12 ACF values.
// Run target: forecast_arima_implementation.
#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
#include <iostream>
#include <sstream>
// Fit ARIMA(1,1,1) on AirPassengers; write the differenced series, the
// sample autocorrelations before and after differencing, and the in-sample
// residuals as CSV plus SVG figures.
namespace {
// Sample autocorrelation r_lag = sum (x_t - m)(x_{t-lag} - m) / sum (x_t - m)^2
// for lag = 1..max_lag (lag 0 is omitted because it is always 1).
// Parameters: x - series; max_lag - number of lags. Returns max_lag values.
// Complexity O(n * max_lag).
ml::Vec autocorrelation(const ml::Vec &x, size_t max_lag) {
  const double avg = ml::mean(x);
  double ss = 0;
  for (double v : x)
    ss += (v - avg) * (v - avg);
  ml::Vec acf;
  for (size_t lag = 1; lag <= max_lag; ++lag) {
    double cov = 0;
    for (size_t i = lag; i < x.size(); ++i)
      cov += (x[i] - avg) * (x[i - lag] - avg);
    acf.push_back(cov / ss);
  }
  return acf;
}
} // namespace
int main() {
  try {
    const auto y = forecast_course::read_passengers(
        std::string(DATA_DIR) + "/air_passengers.csv");
    forecast_course::Arima model(1, 1, 1);
    model.fit(y);
    // w has one element fewer than y: w[t-1] = y[t] - y[t-1].
    const ml::Vec w = forecast_course::Arima::difference(y, 1);
    // RUN_OUTPUT_DIR is injected by CMake (results/03_implementation_results).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    // Table aligned on the original month index t >= 1; residuals[t-1] is the
    // one-step error the model made when w[t-1] arrived.
    std::ostringstream csv;
    csv << "month_index,actual,first_difference,residual\n";
    ml::Vec time, residual;
    for (size_t t = 1; t < y.size(); ++t) {
      time.push_back(double(t));
      residual.push_back(model.residuals[t - 1]);
      csv << t << ',' << y[t] << ',' << w[t - 1] << ',' << residual.back()
          << '\n';
    }
    a.write("differences.csv", csv.str());
    // ACF of the raw series (dominated by the trend) versus the differenced
    // series (dominated by the yearly cycle at lag 12).
    const ml::Vec acf_raw = autocorrelation(y, 24),
                  acf_diff = autocorrelation(w, 24);
    std::ostringstream acf;
    acf << "lag,acf_raw,acf_differenced\n";
    ml::Vec lags;
    for (size_t lag = 1; lag <= 24; ++lag) {
      lags.push_back(double(lag));
      acf << lag << ',' << acf_raw[lag - 1] << ',' << acf_diff[lag - 1] << '\n';
    }
    a.write("autocorrelation.csv", acf.str());
    ml::Plot diff_plot;
    diff_plot.line(time, w, "first difference");
    diff_plot.title("Differenced passengers (d = 1)");
    a.figure("differences.svg", diff_plot);
    ml::Plot acf_plot;
    acf_plot.line(lags, acf_raw, "raw");
    acf_plot.line(lags, acf_diff, "differenced");
    acf_plot.title("Sample autocorrelation");
    a.figure("autocorrelation.svg", acf_plot);
    ml::Plot res_plot;
    res_plot.line(time, residual, "residual");
    a.figure("residuals.svg", res_plot);
    std::cout << "ARIMA(1,1,1): constant " << model.constant << ", phi "
              << model.phi[0] << ", theta " << model.theta[0]
              << "\nLast difference " << w.back() << ", last residual "
              << model.residuals.back() << "\nNext forecast " << model.next()
              << "\nRaw ACF at lag 12: " << acf_raw[11]
              << ", differenced ACF at lag 12: " << acf_diff[11] << "\n";
    return 0;
  } catch (const std::exception &e) {
    // Any I/O or validation error is reported once and mapped to exit code 1.
    std::cerr << e.what() << '\n';
    return 1;
  }
}
