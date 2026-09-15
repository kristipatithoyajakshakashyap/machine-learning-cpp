// 05_bootstrap.cpp
// Lesson: the bootstrap resamples the data (with replacement) many times and
// reads the spread of the resampled means - a CI that needs no normality
// assumption. bootstrap_mean() is seeded, so the same 2000 resamples and the
// same interval come out on every run.
// Equivalent: sklearn.utils.resample / numpy percentile of boot means.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// first 5 boot means: 19.23209 20.18557 20.18619 19.7341 20.12664
// mean of the 2000 boot means: 19.78726
// bootstrap 95% CI [18.68922, 20.99028]  width=2.301057
// normal-ish 95% CI [18.66892, 20.90296]

#include <algorithm>
#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  const double pos = q * static_cast<double>(n - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(n - 1, lo + 1);
  const double frac = pos - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill("total_bill", tips.numeric("total_bill"));

  const std::vector<double> boots = dsts::bootstrap_mean(bill, 2000, 42);
  std::cout << "first 5 boot means:";
  for (size_t i = 0; i < 5; ++i) std::cout << " " << dsts::fmt(boots[i]);
  std::cout << "\n";

  double bsum = 0.0;
  for (double b : boots) bsum += b;
  const double bmean = bsum / static_cast<double>(boots.size());
  std::cout << "mean of the " << boots.size()
            << " boot means: " << dsts::fmt(bmean) << "\n";

  const dsts::ConfidenceInterval boot_ci = {percentile(boots, 0.025),
                                            percentile(boots, 0.975)};
  std::cout << "bootstrap 95% CI [" << dsts::fmt(boot_ci.lower) << ", "
            << dsts::fmt(boot_ci.upper) << "]  width="
            << dsts::fmt(boot_ci.upper - boot_ci.lower) << "\n";

  const dsts::ConfidenceInterval z_ci = dsts::mean_ci95(bill);
  std::cout << "normal-ish 95% CI [" << dsts::fmt(z_ci.lower) << ", "
            << dsts::fmt(z_ci.upper) << "]\n";
  return 0;
}