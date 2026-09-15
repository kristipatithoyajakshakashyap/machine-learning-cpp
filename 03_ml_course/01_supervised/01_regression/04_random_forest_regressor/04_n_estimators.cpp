// 04_n_estimators.cpp
// Lesson: end-to-end, how the number of trees in a random forest affects the
// fit.  Each tree is trained on a bootstrap sample; more trees average away
// variance, so test R2 climbs and then plateaus.  We also check that a deep
// forest can keep improving where a single deep tree overfits.
// Data: california.csv subsampled to 4000 rows for speed.
// sklearn equivalent: sklearn.ensemble.RandomForestRegressor(n_estimators=n).
//
// EXPECTED OUTPUT:
//   california 4000 x 8 (subsample of 20640)
//      1 trees  train R2 0.6463135  test R2 0.4892077
//     50 trees  train R2 0.8239421  test R2 0.716031   <- plateau from here
//    200 trees  train R2 0.8235731  test R2 0.7083637
//   (test R2 climbs then plateaus - averaging trees reduces variance, it
//    never overfits the way growing a single tree deeper does)
//   figure -> results/04_n_estimators_results/n_estimators.svg

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

  auto big = ml::load_california(std::string(DATA_DIR));
  auto ds = ml::subsample(big, 4000, 9);
  std::cout << "california " << ds.n() << " x " << ds.p()
            << " (subsample of " << big.n() << ")\n";

  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  ml::Plot fig(600, 380);
  fig.title("random forest: trees vs R2 (california, 4000 rows)");
  fig.xlabel("n_trees"); fig.ylabel("R2");

  ml::Vec xs, tr, te;
  for (size_t n : {1u, 5u, 10u, 25u, 50u, 100u, 200u}) {
    ml::RandomForest rf(n, 0, 10, 5);
    rf.fit(train.X, train.y);
    double r2tr = ml::r2(train.y, rf.predict(train.X));
    double r2te = ml::r2(test.y, rf.predict(test.X));
    xs.push_back(static_cast<double>(n));
    tr.push_back(r2tr);
    te.push_back(r2te);
    std::cout << std::setw(4) << n << " trees  train R2 " << r2tr
              << "  test R2 " << r2te << "\n";
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