// 08_random_forest.cpp
// Lesson: random-forest regression on a subsample of California housing.
// A forest is a committee of decision trees, each grown on a bootstrap sample
// of the rows and a random subset of the features per split.  Averaging over
// trees lowers variance without increasing bias as much as a single deep tree.
// For regression the default feature subset is p/3; the prediction is the
// mean of tree predictions.  Because bootstrap + feature draws use a seeded
// RNG, the output is deterministic; sklearn's RNG differs, so the comparison
// is tolerance-based (~1e-4 R2).
// Equivalent: sklearn.ensemble.RandomForestRegressor(n_estimators=100,
//             max_depth=10, random_state=seed).
//
// California housing has 20640 rows; we subsample 2048 (seed 7) so the lesson
// runs quickly but the pattern stays visible.
//
// EXPECTED OUTPUT (2048-row subsample, 80/20, seed 7, 100 trees, depth 10):
// forest  train R2 0.8867502  test R2 0.6834607

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "RandomForest.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto full = ml::load_california(DATA_DIR);
  auto data = ml::subsample(full, 2048, 7);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s08_forest.split");

  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::seed_rng(42);
  ml::RandomForest model(100, 0, 10, 1);
  model.fit(tr.X, tr.y);
  auto tr_pred = model.predict(tr.X);
  auto te_pred = model.predict(te.X);
  std::cout << "forest  train R2 " << ml::r2(tr.y, tr_pred) << "  test R2 "
            << ml::r2(te.y, te_pred) << "\n";
  return 0;
}
