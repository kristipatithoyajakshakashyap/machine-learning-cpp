// 04_lasso.cpp
// Lesson: lasso trades the L2 penalty for an L1 one,
//   (1/(2n))||y - Xw||^2 + lambda*||w||_1,
// which drives coefficients exactly to zero. It is fit with coordinate
// descent and soft-thresholding: each coordinate in turn is re-solved while
// the others stay fixed. The intercept is unpenalised.
// Equivalent: sklearn.linear_model.Lasso(alpha=lambda, tol=1e-10) on the
// same rows. We centre the columns first (like sklearn) so the coordinate
// sweeps decouple from the intercept and converge quickly; the threshold
// n*lambda reproduces sklearn's internal alpha*n_samples scaling exactly.
//
// Data: diabetes.csv, 80/20 split (seeded 42; same rows as lesson 01).
//
// EXPECTED OUTPUT (verified vs sklearn Lasso alpha):
// lasso 10   intercept -136.6406624  nonzero features 6/10  R2 train 0.4954524  test 0.3876237
// lasso 100  intercept -51.3157216   nonzero features 5/10  R2 train 0.3681080  test 0.3024447
// lasso 1000 intercept 153.0594901   nonzero features 0/10  R2 train 0.0000000  test -0.003749109
// artifacts -> results/05_lasso_regression_results/{coefficient_path.csv, coefficient_path.svg}

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "LinearRegression.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  std::string dir = DATA_DIR;
  std::cout << std::setprecision(7);
  auto ds = ml::load_diabetes(dir);
  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  for (double lambda : {10.0, 100.0, 1000.0}) {
    ml::LinearRegression m;
    m.fit_lasso(train.X, train.y, lambda, 10000, 1e-6);
    const auto& w = m.coef();
    size_t nz = 0;
    for (size_t j = 1; j < w.size(); ++j)
      if (std::fabs(w[j]) > 1e-6) ++nz;
    auto yte = m.predict(test.X), ytr = m.predict(train.X);
    std::cout << "lasso " << std::left << std::setw(4) << lambda
              << " intercept " << m.intercept() << "  nonzero features "
              << nz << "/10  R2 train " << ml::r2(train.y, ytr) << "  test "
              << ml::r2(test.y, yte) << "\n";
  }

  // Artifacts: the lasso coefficient path.  Sweep lambda on a log grid and
  // record every (unstandardised) coefficient - the plot shows features
  // dropping to exactly zero one by one as the L1 penalty grows.
  const std::vector<double> lambdas = {0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000};
  const size_t p = train.p();
  std::ostringstream csv;
  csv << std::setprecision(10) << "lambda,intercept";
  for (size_t j = 0; j < p; ++j) csv << ",w" << j + 1;
  csv << ",nonzero,r2_train,r2_test\n";
  ml::Vec log_lambda;
  std::vector<ml::Vec> path(p);
  for (double lambda : lambdas) {
    ml::LinearRegression m;
    m.fit_lasso(train.X, train.y, lambda, 10000, 1e-6);
    const auto& w = m.coef();
    size_t nz = 0;
    csv << lambda << "," << m.intercept();
    for (size_t j = 0; j < p; ++j) {
      csv << "," << w[j + 1];
      path[j].push_back(w[j + 1]);
      if (std::fabs(w[j + 1]) > 1e-6) ++nz;
    }
    csv << "," << nz << "," << ml::r2(train.y, m.predict(train.X)) << ","
        << ml::r2(test.y, m.predict(test.X)) << "\n";
    log_lambda.push_back(std::log10(lambda));
  }
  ml::Plot fig(640, 400);
  fig.title("lasso coefficient path (diabetes)");
  fig.xlabel("log10(lambda)"); fig.ylabel("coefficient");
  for (size_t j = 0; j < p; ++j)
    fig.line(log_lambda, path[j], "w" + std::to_string(j + 1));
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  a.write("coefficient_path.csv", csv.str());
  a.figure("coefficient_path.svg", fig);
  std::cout << "wrote " << a.path("coefficient_path.csv") << "\n"
            << "wrote " << a.path("coefficient_path.svg") << "\n";
  return 0;
}
