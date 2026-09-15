// 09_adaboost.cpp
// Lesson: AdaBoost (SAMME) on breast cancer.
// AdaBoost chains "weak learners" (here: depth-1 decision stumps).  Each round
// weights the data so that misclassified rows count more next time, then the
// new stump votes with confidence alpha = ln((1-err)/err) + ln(K-1).  The final
// prediction sums alpha-weighted votes per class.  Weak learners, combined
// carefully, can achieve strong overall accuracy.
// Equivalent: sklearn.ensemble.AdaBoostClassifier(algorithm='SAMME',
//             n_estimators=100, learning_rate=1.0).  Tolerance-based comp.
//
// Breast cancer, 30 features.
// EXPECTED OUTPUT (80/20 seed 7, 100 stumps):
// adaboost  train accuracy 1  test accuracy 0.9649123

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "AdaBoost.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_breast_cancer(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c09_adaboost.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::seed_rng(11);
  ml::AdaBoost model(100, 1.0);
  model.fit(tr.X, tr.y);
  std::cout << "adaboost  train accuracy "
            << ml::accuracy(tr.y, model.predict(tr.X)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(te.X)) << "\n";
  return 0;
}
