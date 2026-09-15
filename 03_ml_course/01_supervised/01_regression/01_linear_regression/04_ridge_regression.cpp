// 03_ridge.cpp
// Lesson: ridge regression adds an L2 penalty to the least-squares objective,
// (1/2)||y - Xw||^2 + lambda*||w||^2, which shrinks the coefficients and
// tames collinearity. The intercept is never penalised. Solution remains
// closed form: (X'X + lambda I) w = X'y.
// Equivalent: sklearn.linear_model.Ridge(alpha=lambda).
//
// Data: diabetes.csv, 80/20 split (seeded 42, identical rows to lesson 01).
//
// EXPECTED OUTPUT (verified vs sklearn Ridge alpha):
// OLS         intercept -347.2497608  R2 train 0.5356466  test 0.4174287
// ridge 1     intercept -326.3816846  R2 train 0.5354695  test 0.4170056
// ridge 100   intercept -148.5449160  R2 train 0.5118525  test 0.3979569
// ridge 1000  intercept -131.3017223  R2 train 0.4964750  test 0.3976479

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "LinearRegression.hpp"
#include "helper/math/matrix.hpp"
#include "helper/math/metrics.hpp"

int main() {
  std::string dir = DATA_DIR;
  std::cout << std::setprecision(7);
  auto ds = ml::load_diabetes(dir);
  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s01_ols.split");
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  ml::LinearRegression m;
  auto report = [&](const std::string& label, double lambda) {
    m.fit_ridge(train.X, train.y, lambda);
    auto ytr = m.predict(train.X), yte = m.predict(test.X);
    std::cout << "ridge " << label << "  intercept " << m.intercept()
              << "  R2 train " << ml::r2(train.y, ytr)
              << "  test " << ml::r2(test.y, yte) << "\n";
  };

  m.fit_ols(train.X, train.y);
  std::cout << "OLS         intercept " << m.intercept() << "  R2 train "
            << ml::r2(train.y, m.predict(train.X)) << "  test "
            << ml::r2(test.y, m.predict(test.X)) << "\n";

  report("1    ", 1.0);
  report("100  ", 100.0);
  report("1000 ", 1000.0);

  return 0;
}
