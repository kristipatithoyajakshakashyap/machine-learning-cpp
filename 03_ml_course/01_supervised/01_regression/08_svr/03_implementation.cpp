// 03_ml_course/01_supervised/01_regression/08_svr/03_implementation.cpp
// Lesson 03: run course::SVR on a tiny linear problem, then fit an RBF SVR to a
// noisy sine curve and record the prediction with its epsilon tube.
//
// Reads:   nothing (60 points generated in-process, std::mt19937 seed 3).
// Writes:  results/03_implementation_results/predictions.csv and predictions.svg
//          (RUN_OUTPUT_DIR is injected by CMake).
// Run:     cmake --build --preset course --target svr_implementation
//          then build/03_ml_course/01_supervised/01_regression/08_svr/
//          svr_implementation
//
// WHAT YOU LEARN:
//   - The dual solver (FISTA + zero-sum bisection) and how b is recovered.
//   - The epsilon tube: points inside it carry no loss and beta_i == 0.
//
// EXPECTED OUTPUT:
//   An explanatory paragraph, predictions on the 4-point toy problem (close to
//   [-3, -1, 3, 5]) and two "wrote <path>" lines.
#include "Model.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
// Artifacts -> results/03_implementation_results/
//   predictions.csv : (x, y, y_hat, tube_upper, tube_lower) on a 1-D curve
//   predictions.svg : scatter of the data, fitted line and epsilon tube
// Toy problem with the default RBF model, just to show the API round trip.
int main() {
  std::cout
      << R"LESSON(FISTA performs a gradient step on the quadratic dual, then soft-thresholds and clips coefficients. Bisection finds a multiplier enforcing the zero-sum constraint. Chosen step size uses a kernel row-sum upper bound. Interior support-vector KKT conditions recover b; the midpoint between the middle epsilon-loss knots minimizes intercept loss when none are interior. This educational solver uses dense O(n^2) memory and finite iterative tolerance; it is not LIBSVM.)LESSON"
      << "\n";
  course::SVR model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";

  // 1-D synthetic curve: y = sin(x) + noise on [-3, 3], RBF kernel fit.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 0.15);
  ml::Mat Xc;
  ml::Vec xc, yc;
  for (int i = 0; i < 60; ++i) {
    const double x = -3.0 + 6.0 * i / 59.0;
    xc.push_back(x);
    Xc.push_back({x});
    yc.push_back(std::sin(x) + noise(rng));
  }
  // C = 10, epsilon = 0.2, gamma = 0.5, up to 3000 FISTA iterations.
  const double eps = 0.2;
  course::SVR svr(10, eps, true, 0.5, 3000);
  svr.fit(Xc, yc);
  ml::Vec pred = svr.predict(Xc), upper, lower;
  std::ostringstream csv;
  csv << std::setprecision(10) << "x,y,y_hat,tube_upper,tube_lower\n";
  for (size_t i = 0; i < xc.size(); ++i) {
    upper.push_back(pred[i] + eps);
    lower.push_back(pred[i] - eps);
    csv << xc[i] << "," << yc[i] << "," << pred[i] << "," << upper[i] << ","
        << lower[i] << "\n";
  }
  ml::Plot fig(640, 400);
  fig.title("SVR (rbf, C=10, eps=0.2) on y = sin(x) + noise");
  fig.xlabel("x"); fig.ylabel("y");
  fig.scatter(xc, yc, "data");
  fig.line(xc, pred, "prediction");
  fig.line(xc, upper, "+epsilon");
  fig.line(xc, lower, "-epsilon");
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  a.write("predictions.csv", csv.str());
  a.figure("predictions.svg", fig);
  std::cout << "wrote " << a.path("predictions.csv") << "\n"
            << "wrote " << a.path("predictions.svg") << "\n";
}
