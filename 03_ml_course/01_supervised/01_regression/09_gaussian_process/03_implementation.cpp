// 03_ml_course/01_supervised/01_regression/09_gaussian_process/03_implementation.cpp
// Lesson 03: fit course::GaussianProcess to 15 noisy sin(x) samples, plot the
// posterior mean with a 2-sigma band, and scan the length scale by marginal
// likelihood.
//
// Reads:   nothing (samples generated in-process, std::mt19937 seed 11).
// Writes:  results/03_implementation_results/posterior.csv, posterior.svg,
//          marginal_likelihood.csv and marginal_likelihood.svg
//          (RUN_OUTPUT_DIR is injected by CMake).
// Run:     cmake --build --preset course --target gp_implementation
//          then build/03_ml_course/01_supervised/01_regression/09_gaussian_process/
//          gp_implementation
//
// WHAT YOU LEARN:
//   - fit() = Cholesky + two triangular solves; predict() and predict_std() reuse
//     the factor, so prediction is O(n) and O(n^2) per query point.
//   - The posterior band is narrow near observations and widens outside [-3, 3].
//   - The length scale with the highest log marginal likelihood is a principled
//     choice without a validation set.
//
// EXPECTED OUTPUT:
//   An explanatory paragraph, "best length scale by marginal likelihood: ..." and
//   one "wrote <path>" line. Exit code 1 with a message on stderr if fit() throws.
#include "Model.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
// Artifacts -> results/03_implementation_results/
//   posterior.csv          : x, true sin(x), posterior mean, std, mean +/- 2 std
//   posterior.svg          : mean and 2-sigma band with the 15 noisy points
//   marginal_likelihood.csv: length_scale, log marginal likelihood
//   marginal_likelihood.svg: the same curve
// The whole lesson is wrapped in try/catch so a numerical failure (for example a
// non-positive-definite K) is reported instead of terminating the process.
int main() {
  try {
    std::cout
        << R"LESSON(fit() builds K = k(X,X) + noise I, factorises it with Cholesky and stores alpha = K^-1 y. predict() is a kernel-weighted sum of alpha; predict_std() subtracts the explained variance |L^-1 k*|^2 from the prior variance. The marginal likelihood falls out of the same factorisation.)LESSON"
        << "\n";
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 0.15);
    ml::Mat X;
    ml::Vec y;
    for (int i = 0; i < 15; ++i) {
      const double x = -3.0 + 6.0 * i / 14.0;
      X.push_back({x});
      y.push_back(std::sin(x) + noise(rng));
    }
    // noise_var is set to the true noise variance (0.15^2), the ideal case.
    course::GaussianProcess gp(1.0, 1.0, 0.15 * 0.15);
    gp.fit(X, y);
    const ml::Vec xs = ml::linspace(-4.0, 4.0, 161);
    ml::Mat Xs;
    for (double v : xs)
      Xs.push_back({v});
    const ml::Vec mean = gp.predict(Xs), sd = gp.predict_std(Xs);
    std::ostringstream csv;
    csv << std::setprecision(10) << "x,true,mean,std,lower,upper\n";
    ml::Vec lower(xs.size()), upper(xs.size()), truth(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      truth[i] = std::sin(xs[i]);
      lower[i] = mean[i] - 2 * sd[i];
      upper[i] = mean[i] + 2 * sd[i];
      csv << xs[i] << ',' << truth[i] << ',' << mean[i] << ',' << sd[i] << ','
          << lower[i] << ',' << upper[i] << '\n';
    }
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("posterior.csv", csv.str());
    ml::Plot fig;
    fig.title("GP posterior on 15 noisy sin(x) samples (length scale 1)");
    fig.xlabel("x");
    fig.ylabel("y");
    fig.line(xs, mean, "posterior mean");
    fig.line(xs, lower, "mean - 2 std");
    fig.line(xs, upper, "mean + 2 std");
    fig.line(xs, truth, "sin(x)");
    ml::Vec px;
    for (const auto &row : X)
      px.push_back(row[0]);
    fig.scatter(px, y, "observations");
    a.figure("posterior.svg", fig);
    // Length-scale scan: refit for each value and keep log p(y | X).
    const ml::Vec scales = {0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0};
    ml::Vec lml;
    std::ostringstream lcsv;
    lcsv << std::setprecision(10) << "length_scale,log_marginal_likelihood\n";
    for (double l : scales) {
      course::GaussianProcess g(l, 1.0, 0.15 * 0.15);
      g.fit(X, y);
      lml.push_back(g.log_marginal_likelihood());
      lcsv << l << ',' << lml.back() << '\n';
    }
    a.write("marginal_likelihood.csv", lcsv.str());
    ml::Plot lp;
    lp.title("Log marginal likelihood versus length scale");
    lp.xlabel("length scale");
    lp.ylabel("log p(y | X)");
    lp.line(scales, lml);
    a.figure("marginal_likelihood.svg", lp);
    size_t best = 0;
    for (size_t i = 1; i < lml.size(); ++i)
      if (lml[i] > lml[best])
        best = i;
    std::cout << std::setprecision(4) << "best length scale by marginal likelihood: "
              << scales[best] << " (LML " << lml[best] << ")\nwrote "
              << a.path("posterior.svg") << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "implementation: " << e.what() << '\n';
    return 1;
  }
}
