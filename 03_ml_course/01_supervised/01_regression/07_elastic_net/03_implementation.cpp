// 03_ml_course/01_supervised/01_regression/07_elastic_net/03_implementation.cpp
// Lesson 03: exercise course::ElasticNet on a tiny exact problem and on a synthetic
// 3-feature dataset, recording how coefficients shrink as alpha grows.
//
// Reads:   nothing (data is generated in-process with std::mt19937 seed 11).
// Writes:  results/03_implementation_results/coefficients.csv and coefficients.svg
//          (RUN_OUTPUT_DIR is injected by CMake).
// Run:     cmake --build --preset course --target elastic_implementation
//          then build/03_ml_course/01_supervised/01_regression/07_elastic_net/
//          elastic_implementation
//
// WHAT YOU LEARN:
//   - Coordinate descent updates one weight at a time using a cached residual.
//   - The l1_ratio x alpha grid shows ridge (0), elastic (0.5) and lasso (1)
//     behaviour side by side; lasso drives the noise feature w3 to exactly 0.
//
// EXPECTED OUTPUT:
//   A short explanatory paragraph, the predictions on the 4-point toy problem
//   (close to [-3, -1, 3, 5]) and two "wrote <path>" lines.
#include "Model.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
// Artifacts -> results/03_implementation_results/
//   coefficients.csv : (l1_ratio, alpha, intercept, w1..w3) grid of fits
//   coefficients.svg : coefficient path vs log10(alpha) at l1_ratio = 0.5
// Toy problem: y = 2x + 1 exactly, so the unpenalised default fit recovers it.
int main() {
  std::cout
      << R"LESSON(Coordinates update a cached residual after centering X and y. A zero-variance column receives zero weight. The model stores coefficients and intercept; the pipeline separately stores fitted preprocessing.)LESSON"
      << "\n";
  course::ElasticNet model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";

  // Synthetic 3-feature problem: x1 informative, x2 a noisy copy of x1
  // (correlated), x3 pure noise.  y = 2*x1 + 1*x2 + noise.
  std::mt19937 rng(11);
  std::normal_distribution<double> gauss(0.0, 1.0);
  ml::Mat Xs;
  ml::Vec ys;
  for (int i = 0; i < 80; ++i) {
    const double x1 = gauss(rng), x2 = x1 + 0.3 * gauss(rng), x3 = gauss(rng);
    Xs.push_back({x1, x2, x3});
    ys.push_back(2 * x1 + x2 + 0.5 * gauss(rng));
  }
  const std::vector<double> ratios = {0.0, 0.5, 1.0};
  const std::vector<double> alphas = {0.01, 0.03, 0.1, 0.3, 1, 3};
  std::ostringstream csv;
  csv << std::setprecision(10) << "l1_ratio,alpha,intercept,w1,w2,w3\n";
  ml::Vec log_alpha;
  std::vector<ml::Vec> path(3);
  // Fit every (l1_ratio, alpha) pair; only the l1_ratio = 0.5 column feeds the plot.
  for (double ratio : ratios)
    for (double alpha : alphas) {
      course::ElasticNet m(alpha, ratio);
      m.fit(Xs, ys);
      csv << ratio << "," << alpha << "," << m.intercept;
      for (double w : m.coef) csv << "," << w;
      csv << "\n";
      if (ratio == 0.5) {
        log_alpha.push_back(std::log10(alpha));
        for (size_t j = 0; j < 3; ++j) path[j].push_back(m.coef[j]);
      }
    }
  ml::Plot fig(640, 400);
  fig.title("elastic net coefficients vs alpha (l1_ratio = 0.5)");
  fig.xlabel("log10(alpha)"); fig.ylabel("coefficient");
  for (size_t j = 0; j < 3; ++j)
    fig.line(log_alpha, path[j], "w" + std::to_string(j + 1));
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  a.write("coefficients.csv", csv.str());
  a.figure("coefficients.svg", fig);
  std::cout << "wrote " << a.path("coefficients.csv") << "\n"
            << "wrote " << a.path("coefficients.svg") << "\n";
}
