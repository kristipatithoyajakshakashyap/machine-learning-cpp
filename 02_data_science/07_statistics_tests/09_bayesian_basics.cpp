// 09_bayesian_basics.cpp
// Lesson: a Bayesian estimate of a rate combines a prior belief with the
// data. For a binomial count the Beta prior is conjugate: prior Beta(a, b)
// plus s successes and f failures gives posterior Beta(a+s, b+f) - no
// sampling needed. The posterior mean and a 95% credible interval come from
// a fine grid over p; next to them the frequentist Wald interval shows how
// close the two views get once n is large.
// Equivalent: scipy.stats.beta(a+s, b+f).mean() / .ppf([0.025, 0.975]).
//
// Data: titanic.csv - Survived by Sex (female 233/314, male 109/577).
//
// EXPECTED OUTPUT:
// prior Beta(1, 1): flat, "any rate is equally plausible"
//
// ## female: n=314 survived=233
//   posterior Beta(234, 82)  mean 0.7405063  mode 0.742
//   95% credible interval (grid) [0.691, 0.787]
//   frequentist Wald 95% CI       [0.6936453, 0.7904312]
//
// ## male: n=577 survived=109
//   posterior Beta(110, 469)  mean 0.1899827  mode 0.189
//   95% credible interval (grid) [0.159, 0.223]
//   frequentist Wald 95% CI       [0.1569686, 0.2208477]
//
// P(rate_female > rate_male) on the grid = 1
// wrote posterior_grid.csv and posterior.svg

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"

namespace {

constexpr size_t kGridPoints = 1001;  // p = 0, 0.001, ..., 1

struct Posterior {
  double alpha, beta;
  std::vector<double> grid, density;  // normalised so density sums to 1
};

// Evaluate Beta(alpha, beta) on the grid using log-densities for stability.
Posterior beta_posterior(double alpha, double beta) {
  Posterior post{alpha, beta, {}, {}};
  double log_max = -1e300;
  for (size_t i = 0; i < kGridPoints; ++i) {
    const double p = static_cast<double>(i) / static_cast<double>(kGridPoints - 1);
    post.grid.push_back(p);
    const double lp = (p <= 0.0 || p >= 1.0) ? -1e300
                                             : (alpha - 1) * std::log(p) + (beta - 1) * std::log1p(-p);
    post.density.push_back(lp);
    log_max = std::max(log_max, lp);
  }
  double total = 0.0;
  for (double& d : post.density) {
    d = std::exp(d - log_max);
    total += d;
  }
  for (double& d : post.density) d /= total;
  return post;
}

// Smallest grid p whose cumulative mass reaches `q`.
double grid_quantile(const Posterior& post, double q) {
  double cum = 0.0;
  for (size_t i = 0; i < post.grid.size(); ++i) {
    cum += post.density[i];
    if (cum >= q) return post.grid[i];
  }
  return post.grid.back();
}

double grid_mode(const Posterior& post) {
  size_t best = 0;
  for (size_t i = 1; i < post.grid.size(); ++i) {
    if (post.density[i] > post.density[best]) best = i;
  }
  return post.grid[best];
}

Posterior report(const std::string& sex, size_t n, size_t survived) {
  const double prior_a = 1.0, prior_b = 1.0;
  const double failed = static_cast<double>(n - survived);
  Posterior post = beta_posterior(prior_a + static_cast<double>(survived), prior_b + failed);
  const double mean = post.alpha / (post.alpha + post.beta);
  const double p_hat = static_cast<double>(survived) / static_cast<double>(n);
  const double se = std::sqrt(p_hat * (1 - p_hat) / static_cast<double>(n));

  std::cout << "\n## " << sex << ": n=" << n << " survived=" << survived << "\n";
  std::cout << "  posterior Beta(" << post.alpha << ", " << post.beta << ")  mean "
            << dsts::fmt(mean) << "  mode " << dsts::fmt(grid_mode(post)) << "\n";
  std::cout << "  95% credible interval (grid) [" << dsts::fmt(grid_quantile(post, 0.025))
            << ", " << dsts::fmt(grid_quantile(post, 0.975)) << "]\n";
  std::cout << "  frequentist Wald 95% CI       [" << dsts::fmt(p_hat - 1.96 * se) << ", "
            << dsts::fmt(p_hat + 1.96 * se) << "]\n";
  return post;
}

}  // namespace

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  const auto& sex = t.strings("Sex");
  const auto& survived = t.numeric("Survived");
  size_t n_f = 0, s_f = 0, n_m = 0, s_m = 0;
  for (size_t r = 0; r < t.rows(); ++r) {
    const bool alive = survived[r].has_value() && *survived[r] == 1.0;
    if (sex[r] == "female") {
      ++n_f;
      s_f += alive;
    } else {
      ++n_m;
      s_m += alive;
    }
  }

  std::cout << "prior Beta(1, 1): flat, \"any rate is equally plausible\"\n";
  const Posterior female = report("female", n_f, s_f);
  const Posterior male = report("male", n_m, s_m);

  // P(p_f > p_m): sum the product of the two grids over the region p_f > p_m.
  double prob_greater = 0.0;
  std::vector<double> male_cum(kGridPoints, 0.0);
  double running = 0.0;
  for (size_t j = 0; j < kGridPoints; ++j) {
    running += male.density[j];
    male_cum[j] = running;
  }
  for (size_t i = 1; i < kGridPoints; ++i) prob_greater += female.density[i] * male_cum[i - 1];
  std::cout << "\nP(rate_female > rate_male) on the grid = " << dsts::fmt(prob_greater) << "\n";
  std::cout << "  interpretation: with a flat prior and this much data the credible\n"
            << "  interval and the Wald CI almost coincide; the Bayesian one is read\n"
            << "  directly as \"95% probability the rate is in here\". The prior matters\n"
            << "  only when n is small - try Beta(1, 1) versus Beta(10, 10) on 10 rows.\n";

  // --- files ---------------------------------------------------------------
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  dsts::DataFrame out;
  out.add_numeric("p", std::vector<dsts::OptD>(female.grid.begin(), female.grid.end()));
  out.add_numeric("density_female",
                  std::vector<dsts::OptD>(female.density.begin(), female.density.end()));
  out.add_numeric("density_male",
                  std::vector<dsts::OptD>(male.density.begin(), male.density.end()));
  dsts::write_csv(std::string(RUN_OUTPUT_DIR) + "/posterior_grid.csv", out);

  // Scatter of both posteriors (every 5th grid point), coloured by sex.
  std::vector<dsts::OptD> xs, ys;
  std::vector<std::string> groups;
  for (const auto* post : {&female, &male}) {
    for (size_t i = 0; i < kGridPoints; i += 5) {
      xs.push_back(post->grid[i]);
      ys.push_back(post->density[i]);
      groups.push_back(post == &female ? "female" : "male");
    }
  }
  dsts::write_svg_scatter(std::string(RUN_OUTPUT_DIR) + "/posterior.svg",
                          "Posterior survival rate by sex (Beta-binomial)", "survival rate p",
                          "posterior mass", dsts::Series("p", xs), dsts::Series("mass", ys),
                          groups);
  std::cout << "wrote posterior_grid.csv and posterior.svg\n";
  return 0;
}
