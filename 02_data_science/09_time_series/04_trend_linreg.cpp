// 04_trend_linreg.cpp
// Lesson: fit a straight line through time (least squares), read off the
// monthly growth rate, then detrend: residuals = observed - fitted. A
// seasonal pattern must survive in the residuals because the trend alone
// cannot explain the August peaks. R2 = corr^2 for a simple regression.
// Equivalent: numpy.polyfit(t, y, 1).
//
// Data: air_passengers.csv (real data, 144 monthly totals).
//
// EXPECTED OUTPUT:
// slope (passengers/month) = 2.657184
// intercept (month 0)      = 90.30996
// growth ~ +31.88621 passengers/year
// R2 (linear fit)          = 0.8536382
// corr(month index, y)     = 0.9239254

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/features.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

int main() {
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const dsts::Series p("passengers", ap.numeric("Passengers"));
  const auto vals = p.values();
  const size_t n = vals.size();

  double mt = 0, my = 0;
  for (size_t i = 0; i < n; ++i) {
    mt += static_cast<double>(i);
    my += *vals[i];
  }
  mt /= static_cast<double>(n);
  my /= static_cast<double>(n);
  double cov = 0, vt = 0;
  for (size_t i = 0; i < n; ++i) {
    const double dt = static_cast<double>(i) - mt;
    cov += dt * (*vals[i] - my);
    vt += dt * dt;
  }
  const double slope = cov / vt;
  const double intercept = my - slope * mt;
  std::vector<dsts::OptD> t(n);
  for (size_t i = 0; i < n; ++i) t[i] = static_cast<double>(i);
  const double corr = dsts::pearson(dsts::Series("t", t), p);
  std::cout << "slope (passengers/month) = " << dsts::fmt(slope) << "\n";
  std::cout << "intercept (month 0)      = " << dsts::fmt(intercept) << "\n";
  std::cout << "growth ~ +" << dsts::fmt(slope * 12) << " passengers/year"
            << "\n";
  std::cout << "R2 (linear fit)          = " << dsts::fmt(corr * corr) << "\n";
  std::cout << "corr(month index, y)     = " << dsts::fmt(corr) << "\n";
  return 0;
}