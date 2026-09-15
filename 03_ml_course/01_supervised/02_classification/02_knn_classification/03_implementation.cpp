// 02_knn_classifier.cpp
// Lesson: k-Nearest Neighbours classification on the iris dataset.
// The classifier stores the training set and labels a test row by majority
// vote among its k nearest training points (Euclidean distance).  With k=1 the
// boundary is jagged (overfits), with larger k it smooths.  We try k in {1, 5,
// 15} and report train/test accuracy.
// Equivalent: sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,
//             algorithm='brute', metric='minkowski', p=2).
// Exact-match verification via the split sidecar.
//
// Iris: 150 rows x 4 features, 3 species.
// EXPECTED OUTPUT (80/20 seed 7):
// k=1  train accuracy 1  test accuracy 0.9333333
// k=5  train accuracy 0.9666667  test accuracy 0.9333333
// k=15 train accuracy 0.975  test accuracy 1

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "KNN.hpp"
#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_iris(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c02_knn.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  for (size_t k : {size_t(1), size_t(5), size_t(15)}) {
    ml::KNNClassifier model(k);
    model.fit(tr.X, tr.y);
    auto tr_pred = model.predict(tr.X);
    auto te_pred = model.predict(te.X);
    std::cout << "k=" << k << "  train accuracy "
              << ml::accuracy(tr.y, tr_pred) << "  test accuracy "
              << ml::accuracy(te.y, te_pred) << "\n";
  }
  return 0;
}
