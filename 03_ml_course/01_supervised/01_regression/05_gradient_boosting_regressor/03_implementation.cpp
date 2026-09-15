// 09_gradient_boosting.cpp
// Lesson: gradient-boosted decision trees for regression on diabetes.
// Boosting adds models sequentially instead of averaging many (bagging):
//   F_0(x) = mean(y);  r_t = y - F_t(x)  (the pseudo-residual);
//   a shallow tree fits r_t;  F_{t+1} = F_t + lr * tree_t(x).
// Each new tree therefore focuses on the residual errors the ensemble still
// makes, and `lr` (shrinkage) slows the learning so later trees use finer
// corrections.  Stronger than a single tree and typically close to (or better
// than) random forest on tabular data.
// Equivalent: sklearn.ensemble.GradientBoostingRegressor(loss='squared_error',
//             learning_rate=0.1, n_estimators=100, max_depth=2, subsample=1.0).
// The comparison is tolerance-based (~1e-4 R2) because sklearn's tree
// internals differ subtly.
//
// EXPECTED OUTPUT (diabetes, 80/20, seed 7, lr 0.1, 100 iters, depth 2):
// gbdt  train R2 0.6959641  test R2 0.5675781

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "GradientBoostedTrees.hpp"
#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_diabetes(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s09_gbdt.split");

  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::GradientBoostedTrees model(100, 0.1, 2, 1, 1.0);
  model.fit(tr.X, tr.y);
  auto tr_pred = model.predict(tr.X);
  auto te_pred = model.predict(te.X);
  std::cout << "gbdt  train R2 " << ml::r2(tr.y, tr_pred) << "  test R2 "
            << ml::r2(te.y, te_pred) << "\n";
  return 0;
}
