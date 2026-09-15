// 08_gradient_boosted_trees.cpp
// Lesson: gradient-boosted trees for classification on the titanic dataset.
// Boosting is additive, not an average: rounds of shallow regression trees are
// fit to the negative gradient of the multinomial log-loss (softmax of K raw
// score functions, one per class).  Each round adds lr * tree_k(x) to score_k,
// so successive trees correct the residual errors of the whole ensemble.  This
// is the algorithm behind most tabular-data competition wins.
// Equivalent: sklearn.ensemble.GradientBoostingClassifier(loss='log_loss',
//             learning_rate=0.1, n_estimators=100, max_depth=2).
// Tolerance-based comparison.
//
// Titanic: 891 rows, 7 engineered features, survived (0/1).
// EXPECTED OUTPUT (80/20 seed 7, 100 rounds, lr 0.1, depth 2):
// gbdt  train accuracy 0.8904494  test accuracy 0.8100559

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "GradientBoostedTrees.hpp"
#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_titanic(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c08_gbdt.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::GradientBoostedTrees model(100, 0.1, 2, 1, 1.0);
    model.set_task(ml::TreeTask::Classification);
  model.fit(tr.X, tr.y);
  std::cout << "gbdt  train accuracy "
            << ml::accuracy(tr.y, model.predict(tr.X)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(te.X)) << "\n";
  return 0;
}
