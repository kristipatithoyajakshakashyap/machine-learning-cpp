// 03_forecast_benchmark.cpp
// Lesson: before fancy models, benchmark against trivial forecasts. Train on
// the first 120 months, test on the last 24: naive (last seen), historical
// mean, and seasonal naive (same month last year). RMSE says who wins.
// Equivalent: pandas.shift() + sklearn.metrics.mean_squared_error.
//
// Data: air_passengers.csv (real data, 144 monthly totals).
//
// EXPECTED OUTPUT:
// test windows: months 120-143
//  RMSE naive            = 137.329
//  RMSE historical mean  = 219.4392
//  RMSE seasonal naive   = 49.98666
//  (seasonal naive wins when the series is strongly seasonal)

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"
#include "dsts/timeseries.hpp"

int main() {
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const auto p = ap.numeric("Passengers");
  const size_t test_start = 120;  // last 24 months are the test set

  // Reference forecasts, NA in the train region.
  std::vector<dsts::OptD> naive(p.size()), mean_f(p.size()), seas(p.size());
  double train_sum = 0.0;
  for (size_t r = 0; r < test_start; ++r) train_sum += *p[r];
  const double train_mean = train_sum / static_cast<double>(test_start);
  for (size_t r = 0; r < p.size(); ++r) {
    if (r < test_start) {
      naive[r] = mean_f[r] = seas[r] = std::nullopt;
    } else {
      naive[r] = *p[test_start - 1];  // repeat the last train value
      mean_f[r] = train_mean;         // flat historical mean
      seas[r] = *p[r - 12];           // same month a year earlier
    }
  }

  const dsts::Series actual("actual", p);
  std::cout << "test windows: months " << 120 << "-" << p.size() - 1 << "\n";
  std::cout << " RMSE naive            = "
            << dsts::fmt(dsts::rmse(actual, dsts::Series("naive", naive))) << "\n";
  std::cout << " RMSE historical mean  = "
            << dsts::fmt(dsts::rmse(actual, dsts::Series("mean", mean_f))) << "\n";
  std::cout << " RMSE seasonal naive   = "
            << dsts::fmt(dsts::rmse(actual, dsts::Series("seasonal", seas)))
            << "\n";
  std::cout << " (seasonal naive wins when the series is strongly seasonal)\n";
  return 0;
}