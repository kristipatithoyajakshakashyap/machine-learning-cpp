// 03_decision_tree_classifier.cpp
// Lesson: CART decision-tree classification on the penguins dataset.
// The tree partitions the feature space with the same split-finder as the
// regression tree, but measuring impurity with Gini (1 - sum p_k^2) instead
// of MSE.  Each leaf stores the empirical class distribution; a test row is
// classified by the majority class of the leaf it lands in.  max_depth and
// min_samples_leaf bound the tree so it does not memorise the training set.
// Equivalent: sklearn.tree.DecisionTreeClassifier(criterion='gini',
//             max_depth=4, min_samples_leaf=5).
// Exact-match verified against sklearn on the same split sidecar.
//
// Penguins: 342 rows x 4 measurements, 3 species.
// EXPECTED OUTPUT (80/20 seed 7, depth 4, leaf 5):
// nodes 15  train accuracy 0.970696  test accuracy 1

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "DecisionTree.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_penguins(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c03_tree.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::DecisionTree model(ml::TreeTask::Classification,
                         ml::SplitCriterion::Gini, 4, 5);
  model.fit(tr.X, tr.y);
  auto tr_pred = model.predict(tr.X);
  auto te_pred = model.predict(te.X);
  std::cout << "nodes " << model.node_count() << "  train accuracy "
            << ml::accuracy(tr.y, tr_pred) << "  test accuracy "
            << ml::accuracy(te.y, te_pred) << "\n";
  return 0;
}
