#pragma once
// Shared model factory for this module: the course random forest configured
// as a classifier.  The algorithm itself is compiled once in ml_tree_models.
#include "01_supervised/01_regression/04_random_forest_regressor/RandomForest.hpp"
#include "helper/math/optim.hpp"
#include <cstddef>

namespace course {
inline ml::RandomForest make_forest(size_t n_trees, size_t max_depth = 8,
                                    size_t min_leaf = 2) {
  ml::RandomForest forest(n_trees, 0, max_depth, min_leaf);
  forest.set_task(ml::TreeTask::Classification);
  return forest;
}
} // namespace course
