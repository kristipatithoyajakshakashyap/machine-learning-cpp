// 02_gradient_descent.cpp
// Lesson: instead of solving (X'X)w = X'y in one shot, we can walk downhill.
// Full-batch gradient descent and Adam minimise the same MSE objective; the
// closed-form OLS answer is the target they converge to. This lesson also
// verifies the analytic gradient against a finite-difference check.
// Equivalent: the closed form matches sklearn.linear_model.LinearRegression
// on the same single feature exactly; the optimisers reproduce it by
// iteration (Adam follows the Kingma & Ba update with a fixed seed).
//
// Data: tips.csv (244 restaurant bills; total_bill -> tip). Only the
// total_bill column feeds the model so all three fits share one target.
//
// EXPECTED OUTPUT:
// closed-form OLS: intercept 0.9202696  total_bill 0.1050245  R2 0.4566166
// GD 1000 iters lr=1e-5: intercept 0.009094428  total_bill 0.1433519
// Adam 2000 iters lr=0.2: intercept 0.9203163  total_bill 0.1050244
// gradient check at w0: max|fd-analytic| = 4.615084e-09

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "helper/data/datasets.hpp"
#include "LinearRegression.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

namespace {

// single-feature design: bias column first, then total_bill (col 0).
ml::Mat design(const ml::Dataset& d) {
  ml::Mat M(d.n(), ml::Vec(2, 1.0));
  for (size_t i = 0; i < d.n(); ++i) M[i][1] = d.X[i][0];
  return M;
}

// single-column view of X (used for the sklearn-equivalent closed form).
ml::Mat one_feature(const ml::Dataset& d) {
  ml::Mat M(d.n(), ml::Vec(1));
  for (size_t i = 0; i < d.n(); ++i) M[i][0] = d.X[i][0];
  return M;
}

// MSE objective and its analytic gradient (2/n) M'(Mw - y).
double mse(const ml::Mat& M, const ml::Vec& y, const ml::Vec& w) {
  double s = 0.0;
  for (size_t i = 0; i < M.size(); ++i) {
    double pred = w[0];
    for (size_t j = 1; j < w.size(); ++j) pred += w[j] * M[i][j];
    double e = pred - y[i];
    s += e * e;
  }
  return s / static_cast<double>(M.size());
}

void gradf(const ml::Mat& M, const ml::Vec& y, const ml::Vec& w,
           ml::Vec& g) {
  g.assign(w.size(), 0.0);
  double n = static_cast<double>(M.size());
  for (size_t i = 0; i < M.size(); ++i) {
    double pred = w[0];
    for (size_t j = 1; j < w.size(); ++j) pred += w[j] * M[i][j];
    double e = pred - y[i];
    for (size_t j = 0; j < w.size(); ++j) g[j] += e * M[i][j];
  }
  for (double& v : g) v *= 2.0 / n;
}

}  // namespace

int main() {
  std::string dir = DATA_DIR;
  std::cout << std::setprecision(7);
  auto ds = ml::load_tips(dir);
  auto M = design(ds);
  auto Xf = one_feature(ds);

  ml::LinearRegression ols;
  ols.fit_ols(Xf, ds.y);
  std::cout << "closed-form OLS: intercept " << ols.intercept()
            << "  total_bill " << ols.coef()[1] << "  R2 "
            << ml::r2(ds.y, ols.predict(Xf)) << "\n";

  auto loss = [&](const ml::Vec& w) { return mse(M, ds.y, w); };
  auto grad = [&](const ml::Vec& w, ml::Vec& g) { gradf(M, ds.y, w, g); };

  ml::seed_rng(1);
  ml::Vec w0(2, 0.0);
  auto gd = ml::gradient_descent(w0, 1000, 1e-5, loss, grad);
  std::cout << "GD 1000 iters lr=1e-5: intercept " << gd.w[0]
            << "  total_bill " << gd.w[1] << "\n";

  auto adam = ml::adam(w0, 2000, 0.2, loss, grad);
  std::cout << "Adam 2000 iters lr=0.2: intercept " << adam.w[0]
            << "  total_bill " << adam.w[1] << "\n";

  std::cout << "gradient check at w0: max|fd-analytic| = "
            << ml::gradient_check(loss, grad, w0) << "\n";

  return 0;
}
