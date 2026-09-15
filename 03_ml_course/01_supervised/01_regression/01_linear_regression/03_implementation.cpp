// 01_ols.cpp
// Lesson: ordinary least squares (closed form via normal equations). The
// weights are computed in one shot as w = (X'X)^{-1} X'y and never
// touched by an iterative optimiser.
// Equivalent: sklearn.linear_model.LinearRegression (fit_intercept=True,
// solver='auto'). With normal equations + partial-pivot elimination the
// numbers match sklearn to 7+ digits.
//
// Data: diabetes.csv (442 patients, 10 features, regression target).
//
// EXPECTED OUTPUT:
// features 442 x 10  target mean 152.1335
// intercept -347.2497608  age 0.0544782  sex -21.0310746  bmi 6.3090619  bp 1.1364427  s1 -0.9609946  s2 0.7087509  s3 0.2944921  s4 1.7277955  s5 67.9370070  s6 0.2022571
// train R2 0.5356466  RMSE 52.77683
// test  R2 0.4174287  RMSE 57.32665

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
  std::cout << "features " << ds.n() << " x " << ds.p()
            << "  target mean " << ml::mean(ds.y) << "\n";

  // deterministic 80/20 split
  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s01_ols.split");
  auto train = ml::select_rows(ds, fold.train);
  auto test  = ml::select_rows(ds, fold.test);

  ml::LinearRegression lr;
  lr.fit_ols(train.X, train.y);

  std::cout << ml::print_coefs(lr, ds.feature_names) << "\n";

  auto yhat_train = lr.predict(train.X);
  auto yhat_test  = lr.predict(test.X);

  std::cout << "train R2 " << ml::r2(train.y, yhat_train)
            << "  RMSE " << ml::rmse(train.y, yhat_train) << "\n";
  std::cout << "test  R2 " << ml::r2(test.y, yhat_test)
            << "  RMSE " << ml::rmse(test.y, yhat_test) << "\n";

  return 0;
}
