// 04_boot_sample_ci.cpp
// Lesson: sample_indices() draws with a seed and without replacement - the
// building block both for splits and for the bootstrap. Resampling the whole
// tips table (with replacement, bootstrap style) puts a confidence interval
// around the bill-tip correlation.
// Equivalent: numpy.random.default_rng(seed).choice / .permutation.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// pearson(total_bill, tip) = 0.6757341
// sample_indices(20,6,99) twice: same -> 3 10 13 14 16 17
// bootstrap 95% CI for correlation [0.5754407, 0.758738]

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"
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
  const dsts::Series bill("b", tips.numeric("total_bill"));
  const dsts::Series tip("t", tips.numeric("tip"));
  std::cout << "pearson(total_bill, tip) = "
            << dsts::fmt(dsts::pearson(bill, tip)) << "\n";

  const auto a = dsts::sample_indices(20, 6, 99);
  const auto b = dsts::sample_indices(20, 6, 99);
  std::cout << "sample_indices(20,6,99) twice: " << (a == b ? "same" : "diff")
            << " ->";
  for (size_t i : a) std::cout << " " << i;
  std::cout << "\n";

  // Bootstrap the correlation over 2000 resamples, seeded for reproducibility.
  const auto& bill_v = bill.values();
  const auto& tip_v = tip.values();
  std::mt19937 rng(7u);
  std::vector<double> boots;
  boots.reserve(2000);
  for (int iter = 0; iter < 2000; ++iter) {
    std::vector<dsts::OptD> x, y;
    for (size_t i = 0; i < bill_v.size(); ++i) {
      std::uniform_int_distribution<size_t> d(0, bill_v.size() - 1);
      const size_t j = d(rng);
      x.push_back(bill_v[j]);
      y.push_back(tip_v[j]);
    }
    boots.push_back(dsts::pearson(dsts::Series("x", x), dsts::Series("y", y)));
  }
  const double lo = percentile(boots, 0.025), hi = percentile(boots, 0.975);
  std::cout << "bootstrap 95% CI for correlation [" << dsts::fmt(lo) << ", "
            << dsts::fmt(hi) << "]\n";
  return 0;
}