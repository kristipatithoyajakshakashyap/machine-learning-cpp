// 08_effect_sizes_multiple_testing.cpp
// Lesson: a p-value says "is there a difference?"; an effect size says "how
// big?". Cohen's d measures a mean shift in sd units, Cliff's delta the
// probability that one group beats the other. Running six pairwise tests at
// alpha = 0.05 inflates the family-wise error rate, so Bonferroni and Holm
// adjust the p-values. A seeded simulation shows the t-test's power (its
// chance of catching a real effect) growing with sample size.
// Equivalent: statsmodels.stats.multitest.multipletests(method="bonferroni"
// / "holm"), pingouin.compute_effsize, statsmodels TTestIndPower.
//
// Data: tips.csv (244 tips) - tip by day, all six day pairs.
//
// EXPECTED OUTPUT:
// ## pairwise Welch t-tests: tip by day (6 pairs, alpha 0.05)
//   pair        d            cliff        p_raw        bonferroni   holm
//   Thur-Fri  0.03076116   -0.0475382   0.8971389    1            1
//   Thur-Sat  -0.1496216   -0.07804968  0.3478332    1            1
//   Thur-Sun  -0.3909221   -0.2548812   0.02403244   0.1441946    0.1441946
//   Fri-Sat   -0.1674845   -0.02238355  0.3814787    1            1
//   Fri-Sun   -0.4350265   -0.2610803   0.06590542   0.3954325    0.3295271
//   Sat-Sun   -0.1794818   -0.1978221   0.2460283    1            0.9841131
// rejected: raw 1, bonferroni 0, holm 0
//
// ## power of the Welch t-test for d = 0.5 (500 seeded sims per n)
//   n=10 power=0.188 n=20 power=0.362 n=30 power=0.508 ... n=100 power=0.944
//
// wrote corrections.csv and power_curve.svg

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

namespace {

std::vector<double> group_values(const dsts::DataFrame& df, const std::string& level) {
  std::vector<double> out;
  const auto& day = df.strings("day");
  const auto& tip = df.numeric("tip");
  for (size_t r = 0; r < df.rows(); ++r) {
    if (day[r] == level && tip[r].has_value()) out.push_back(*tip[r]);
  }
  return out;
}

double mean_of(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

double var_of(const std::vector<double>& v, double m) {
  double s = 0.0;
  for (double x : v) s += (x - m) * (x - m);
  return s / static_cast<double>(v.size() - 1);
}

// Cohen's d with the pooled standard deviation.
double cohens_d(const std::vector<double>& a, const std::vector<double>& b) {
  const double ma = mean_of(a), mb = mean_of(b);
  const double na = static_cast<double>(a.size()), nb = static_cast<double>(b.size());
  const double pooled =
      std::sqrt(((na - 1) * var_of(a, ma) + (nb - 1) * var_of(b, mb)) / (na + nb - 2));
  return (ma - mb) / pooled;
}

// Cliff's delta: P(a > b) - P(a < b) over all pairs, in [-1, 1].
double cliffs_delta(const std::vector<double>& a, const std::vector<double>& b) {
  long long score = 0;
  for (double x : a) {
    for (double y : b) score += (x > y) - (x < y);
  }
  return static_cast<double>(score) / static_cast<double>(a.size() * b.size());
}

std::vector<double> bonferroni(const std::vector<double>& p) {
  std::vector<double> out;
  for (double v : p) out.push_back(std::min(1.0, v * static_cast<double>(p.size())));
  return out;
}

// Holm step-down: sort p ascending, multiply the i-th by (m - i), enforce
// monotonicity, then put the values back in the original order.
std::vector<double> holm(const std::vector<double>& p) {
  const size_t m = p.size();
  std::vector<size_t> order(m);
  for (size_t i = 0; i < m; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p[a] < p[b]; });
  std::vector<double> out(m);
  double running = 0.0;
  for (size_t i = 0; i < m; ++i) {
    const double adj = std::min(1.0, p[order[i]] * static_cast<double>(m - i));
    running = std::max(running, adj);
    out[order[i]] = running;
  }
  return out;
}

dsts::Series to_series(const std::string& name, const std::vector<double>& v) {
  return dsts::Series(name, std::vector<dsts::OptD>(v.begin(), v.end()));
}

// Fraction of seeded simulations in which Welch's t rejects at alpha 0.05
// when the true shift is `d` standard deviations and each group has n rows.
double simulated_power(size_t n, double d, size_t sims, std::mt19937& rng) {
  std::normal_distribution<double> control(0.0, 1.0), treated(d, 1.0);
  size_t rejected = 0;
  for (size_t s = 0; s < sims; ++s) {
    std::vector<double> a(n), b(n);
    for (double& x : a) x = control(rng);
    for (double& x : b) x = treated(rng);
    if (dsts::ttest_ind(to_series("a", a), to_series("b", b)).p_value < 0.05) ++rejected;
  }
  return static_cast<double>(rejected) / static_cast<double>(sims);
}

}  // namespace

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const std::vector<std::string> days{"Thur", "Fri", "Sat", "Sun"};
  std::vector<std::vector<double>> groups;
  for (const auto& d : days) groups.push_back(group_values(tips, d));

  std::vector<std::string> pair_names;
  std::vector<double> d_vals, cliff_vals, p_raw;
  for (size_t i = 0; i < days.size(); ++i) {
    for (size_t j = i + 1; j < days.size(); ++j) {
      pair_names.push_back(days[i] + "-" + days[j]);
      d_vals.push_back(cohens_d(groups[i], groups[j]));
      cliff_vals.push_back(cliffs_delta(groups[i], groups[j]));
      p_raw.push_back(dsts::ttest_ind(to_series("a", groups[i]), to_series("b", groups[j])).p_value);
    }
  }
  const std::vector<double> p_bonf = bonferroni(p_raw);
  const std::vector<double> p_holm = holm(p_raw);

  std::cout << "## pairwise Welch t-tests: tip by day (" << pair_names.size()
            << " pairs, alpha 0.05)\n";
  std::cout << "  pair        d          cliff      p_raw      bonferroni  holm\n";
  size_t rej_raw = 0, rej_bonf = 0, rej_holm = 0;
  for (size_t i = 0; i < pair_names.size(); ++i) {
    std::cout << "  " << std::left << std::setw(10) << pair_names[i] << std::setw(13)
              << dsts::fmt(d_vals[i]) << std::setw(13) << dsts::fmt(cliff_vals[i])
              << std::setw(13) << dsts::fmt(p_raw[i]) << std::setw(13) << dsts::fmt(p_bonf[i])
              << dsts::fmt(p_holm[i]) << "\n";
    rej_raw += p_raw[i] < 0.05;
    rej_bonf += p_bonf[i] < 0.05;
    rej_holm += p_holm[i] < 0.05;
  }
  std::cout << "rejected: raw " << rej_raw << ", bonferroni " << rej_bonf << ", holm "
            << rej_holm << "\n";
  std::cout << "  Cohen's d: |d| ~0.2 small, ~0.5 medium, ~0.8 large. Cliff's delta\n"
            << "  reads as P(a > b) - P(a < b) and ignores the tip outliers.\n"
            << "  Holm is never more conservative than Bonferroni and controls the\n"
            << "  same family-wise error rate, so prefer it.\n";

  // --- power curve -------------------------------------------------------
  const double effect = 0.5;
  const size_t sims = 500;
  std::mt19937 rng(2024);
  std::vector<std::string> n_labels;
  std::vector<double> power;
  std::cout << "\n## power of the Welch t-test for d = " << effect << " (" << sims
            << " seeded sims per n)\n ";
  for (size_t n = 10; n <= 100; n += 10) {
    power.push_back(simulated_power(n, effect, sims, rng));
    n_labels.push_back(std::to_string(n));
    std::cout << " n=" << n << " power=" << dsts::fmt(power.back());
  }
  std::cout << "\n  power 0.8 is the usual target: for a medium effect that needs\n"
            << "  roughly 64 per group; at n=10 most real effects go undetected.\n";

  // --- files ---------------------------------------------------------------
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  dsts::DataFrame out;
  out.add_string("pair", pair_names);
  out.add_numeric("cohens_d", std::vector<dsts::OptD>(d_vals.begin(), d_vals.end()));
  out.add_numeric("cliffs_delta", std::vector<dsts::OptD>(cliff_vals.begin(), cliff_vals.end()));
  out.add_numeric("p_raw", std::vector<dsts::OptD>(p_raw.begin(), p_raw.end()));
  out.add_numeric("p_bonferroni", std::vector<dsts::OptD>(p_bonf.begin(), p_bonf.end()));
  out.add_numeric("p_holm", std::vector<dsts::OptD>(p_holm.begin(), p_holm.end()));
  dsts::write_csv(std::string(RUN_OUTPUT_DIR) + "/corrections.csv", out);
  dsts::write_svg_line(std::string(RUN_OUTPUT_DIR) + "/power_curve.svg",
                       "Welch t-test power, d = 0.5", "n per group", "power", n_labels,
                       to_series("power", power));
  std::cout << "\nwrote corrections.csv and power_curve.svg\n";
  return 0;
}
