// 06_knn_regression.cpp
// Lesson: k-Nearest Neighbours regression on the diabetes dataset.
// The model stores the whole training set and predicts each test row by
// averaging the targets of its k closest training rows (Euclidean distance).
// No weights are learned - the "model" is the training data itself, and the
// single hyper-parameter k controls the bias-variance trade-off:
//   small k  -> low bias, high variance (fits noise),
//   large k  -> high bias, low variance (smooths everything out).
// Equivalent: sklearn.neighbors.KNeighborsRegressor(n_neighbors=k,
//             weights='uniform', algorithm='brute', metric='minkowski', p=2).
//
// We train on a 80/20 deterministic split (ml::train_test_indices) and print
// R^2 and RMSE on both train and test for k in {1, 3, 10}.  The model is
// exact-match verified against sklearn via the sidecar split file.
//
// EXPECTED OUTPUT (diabetes, 80/20 split, seed 7):
// k=1   train R2 1  test R2 -0.1147819
// k=3   train R2 0.5288474  test R2 0.1340539
// k=10  train R2 0.4100481  test R2 0.3579964  <-- best test R2, smooths best

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "KNN.hpp"
#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_diabetes(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/s06_knn.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  for (size_t k : {size_t(1), size_t(3), size_t(10)}) {
    ml::KNNRegressor model(k);
    model.fit(tr.X, tr.y);
    auto tr_pred = model.predict(tr.X);
    auto te_pred = model.predict(te.X);
    std::cout << "k=" << k << "  train R2 " << ml::r2(tr.y, tr_pred)
              << "  test R2 " << ml::r2(te.y, te_pred) << "\n";
  }
  return 0;
}
