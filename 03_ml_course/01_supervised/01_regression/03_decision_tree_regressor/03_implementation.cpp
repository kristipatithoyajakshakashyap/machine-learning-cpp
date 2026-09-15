// 07_decision_tree.cpp
// Lesson: decision-tree regression on the diabetes dataset.
// A CART tree recursively splits the feature space: at each node it picks the
// feature and threshold that most reduces the weighted squared-error (MSE)
// impurity of the two children, and repeats until max_depth / min_samples_leaf
// stop further splits.  Prediction for a new row = mean target of the training
// rows in the leaf it falls into.
// Equivalent: sklearn.tree.DecisionTreeRegressor(criterion='squared_error',
//             max_depth=4, min_samples_leaf=5).
//
// The splitter uses exactly the same rule as sklearn (midpoint thresholds
// between distinct sorted values, first-best tie-break), so on the same split
// sidecar predictions match digit-for-digit.
//
// EXPECTED OUTPUT (diabetes, 80/20 split, seed 7, depth 4, leaf 5):
// depth=4 leaf=5  nodes 25  train R2 0.5697488  test R2 0.4538644
// depth=8 leaf=5  nodes 111  train R2 0.7982585  test R2 0.2356813

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "DecisionTree.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_diabetes(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s07_tree.split");

  for (size_t depth : {size_t(4), size_t(8)}) {
    auto tr = ml::select_rows(data, fold.train);
    auto te = ml::select_rows(data, fold.test);
    ml::DecisionTree model(ml::TreeTask::Regression, ml::SplitCriterion::Gini,
                           depth, 5);
    model.fit(tr.X, tr.y);
    auto tr_pred = model.predict(tr.X);
    auto te_pred = model.predict(te.X);
    std::cout << "depth=" << depth << " leaf=5  nodes " << model.node_count()
              << "  train R2 " << ml::r2(tr.y, tr_pred) << "  test R2 "
              << ml::r2(te.y, te_pred) << "\n";
  }
  return 0;
}
