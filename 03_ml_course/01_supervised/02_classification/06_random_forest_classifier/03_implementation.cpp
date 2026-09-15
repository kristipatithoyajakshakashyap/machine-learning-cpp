// 07_random_forest_classifier.cpp
// Lesson: random-forest classification on the penguins dataset.
// Each of 100 trees grows on a bootstrap sample of the rows plus a random
// sqrt(p) feature subset per split (sklearn's default).  Classification is a
// soft vote: average the per-class probabilities over all trees, then take
// the argmax.  This reduces variance dramatically vs a single tree.
// Equivalent: sklearn.ensemble.RandomForestClassifier(n_estimators=100,
//             max_depth=10, random_state=seed).  Tolerance-based comparison.
//
// Penguins: 342 rows, 3 classes.
// EXPECTED OUTPUT (80/20 seed 7, 100 trees, depth 10):
// forest  train accuracy 1  test accuracy 1

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "RandomForest.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_penguins(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c07_forest.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::seed_rng(42);
  ml::RandomForest model(100, 0, 10, 1);
    model.set_task(ml::TreeTask::Classification);  // max_features 0 => sqrt(p)
  model.fit(tr.X, tr.y);
  std::cout << "forest  train accuracy "
            << ml::accuracy(tr.y, model.predict(tr.X)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(te.X)) << "\n";
  return 0;
}
