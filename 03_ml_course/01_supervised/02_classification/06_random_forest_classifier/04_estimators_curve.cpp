// 04_estimators_curve.cpp
// Lesson: end-to-end, how many trees does a random forest classifier need?
// Unlike a single tree (which overfits as it grows), a forest only gets
// better as trees are added, because bootstrap averaging reduces variance.
// The curve shows train accuracy ~1.0 from the start while test accuracy
// climbs before plateauing - diminishing returns, not overfitting.
// Data: breast_cancer.csv (569 x 30).  sklearn equivalent:
// sklearn.ensemble.RandomForestClassifier(n_estimators=n).
//
// EXPECTED OUTPUT:
//   breast_cancer 569 x 30
//      1 trees  train acc 0.9736264  test acc 0.9210526
//     50 trees  train acc 1  test acc 0.9649123
//    200 trees  train acc 1  test acc 0.9736842
//   (train accuracy is ~1.0 from a handful of trees; test accuracy climbs
//    with more trees - diminishing returns, not overfitting)
//   figure -> results/04_estimators_curve_results/n_estimators.svg

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"
#include "RandomForest.hpp"

int main() {
  std::cout << std::setprecision(7);

  auto ds = ml::load_breast_cancer(std::string(DATA_DIR));
  std::cout << "breast_cancer " << ds.n() << " x " << ds.p() << "\n";

  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  ml::Plot fig(600, 380);
  fig.title("random forest: trees vs accuracy (breast cancer)");
  fig.xlabel("n_trees"); fig.ylabel("accuracy");

  ml::Vec xs, tr, te;
  for (size_t n : {1u, 5u, 10u, 25u, 50u, 100u, 200u}) {
    ml::RandomForest rf(n, 0, 10, 1);
    rf.set_task(ml::TreeTask::Classification);
    rf.fit(train.X, train.y);
    double acc_tr = ml::accuracy(train.y, rf.predict(train.X));
    double acc_te = ml::accuracy(test.y, rf.predict(test.X));
    xs.push_back(static_cast<double>(n));
    tr.push_back(acc_tr);
    te.push_back(acc_te);
    std::cout << std::setw(4) << n << " trees  train acc " << acc_tr
              << "  test acc " << acc_te << "\n";
  }

  fig.line(xs, tr, "train");
  fig.line(xs, te, "test");

  std::string out = std::string(RUN_OUTPUT_DIR) + "/n_estimators.svg";
  std::ofstream f(out);
  f << fig.render();
  f.close();
  std::cout << "figure -> " << out << "\n";
  return 0;
}