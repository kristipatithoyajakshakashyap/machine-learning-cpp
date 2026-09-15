// 05_polynomial.cpp
// Lesson: a linear model on polynomial features fits curves. With degree d
// the design matrix gains every monomial of total degree <= d; features are
// standardised (train-only mean/std) before fitting so the solver is stable.
// Watching train R2 climb while test R2 peaks then falls is the classic
// overfitting picture.
// Equivalent: sklearn.pipeline make_pipeline(PolynomialFeatures,
// StandardScaler, LinearRegression) - R2/RMSE match exactly (our monomial
// column order differs, predictions do not).
//
// Data: tips.csv (total_bill -> tip), 80/20 split seeded 42.
// Only column 0 (total_bill) feeds the polynomial; column 1 (size) is unused.
//
// EXPECTED OUTPUT (verified vs sklearn via same-split row indices):
// degree 1  train R2 0.4455523  test R2 0.4373418  train RMSE 1.04876  test RMSE 0.8965347
// degree 2  train R2 0.4461932  test R2 0.4267559  train RMSE 1.048153  test RMSE 0.9049291
// degree 7  train R2 0.5295192  test R2 0.3966204  train RMSE 0.966088  test RMSE 0.9284105

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
  auto ds = ml::load_tips(dir);
  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s05_poly.split");
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  // keep only the single feature total_bill (column 0)
  auto one_col = [](const ml::Dataset& d) {
    ml::Mat m{d.n(), ml::Vec(1)};
    for (size_t i = 0; i < d.n(); ++i) m[i][0] = d.X[i][0];
    return m;
  };
  ml::Mat Xtr0 = one_col(train), Xte0 = one_col(test);

  for (size_t deg : {size_t(1), size_t(2), size_t(7)}) {
    ml::PolynomialFeatures poly(deg, /*include_bias=*/false);
    auto Xtr = poly.transform(Xtr0);
    auto Xte = poly.transform(Xte0);

    ml::Vec mu, sd;
    auto Ztr = ml::scale_fit(Xtr, &mu, &sd);
    auto Zte = ml::scale_apply(Xte, mu, sd);

    ml::LinearRegression m;
    m.fit_ols(Ztr, train.y);
    auto ytr = m.predict(Ztr), yte = m.predict(Zte);

    std::cout << "degree " << deg << "  train R2 " << ml::r2(train.y, ytr)
              << "  test R2 " << ml::r2(test.y, yte) << "  train RMSE "
              << ml::rmse(train.y, ytr) << "  test RMSE "
              << ml::rmse(test.y, yte) << "\n";
  }
  return 0;
}
