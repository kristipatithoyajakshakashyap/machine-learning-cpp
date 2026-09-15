// 04_depth_curve.cpp
// Lesson: end-to-end, how decision-tree depth trades bias against variance.
// Deep trees memorise the training set (train R2 -> 1) but generalise worse;
// shallow trees underfit.  We sweep max_depth and plot train/test R2 so the
// sweet spot is visible.
// Data: california.csv subsampled to 4000 rows for speed.
// sklearn equivalent: sklearn.tree.DecisionTreeRegressor(max_depth=d).
//
// EXPECTED OUTPUT:
//   california 4000 x 8 (subsample of 20640)
//   depth  1  train R2 0.3319885  test R2 0.3187464
//   depth  5  train R2 0.6285316  test R2 0.6062116
//   depth 10  train R2 0.8483963  test R2 0.6913839  <- best
//   depth 16  train R2 0.8834439  test R2 0.6751301  (overfit: deep trees
//             memorise train, test R2 turns back down)
//   best test R2 at depth 10
//   figure -> results/04_depth_curve_results/depth_curve.svg

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"
#include "DecisionTree.hpp"

int main() {
  std::cout << std::setprecision(7);

  auto big = ml::load_california(std::string(DATA_DIR));
  auto ds = ml::subsample(big, 4000, 7);
  std::cout << "california " << ds.n() << " x " << ds.p()
            << " (subsample of " << big.n() << ")\n";

  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  ml::Plot fig(600, 380);
  fig.title("decision tree depth vs R2 (california, 4000 rows)");
  fig.xlabel("max_depth"); fig.ylabel("R2");

  ml::Vec xs, tr, te;
  double best_r2 = -1e300;
  size_t best_d = 0;
  for (size_t d = 1; d <= 16; ++d) {
    ml::DecisionTree tree(ml::TreeTask::Regression, ml::SplitCriterion::Gini,
                          d, 5);
    tree.fit(train.X, train.y);
    double r2tr = ml::r2(train.y, tree.predict(train.X));
    double r2te = ml::r2(test.y, tree.predict(test.X));
    xs.push_back(static_cast<double>(d));
    tr.push_back(r2tr);
    te.push_back(r2te);
    std::cout << "depth " << std::setw(2) << d << "  train R2 " << r2tr
              << "  test R2 " << r2te << "\n";
    if (r2te > best_r2) { best_r2 = r2te; best_d = d; }
  }
  std::cout << "best test R2 at depth " << best_d << "\n";

  fig.line(xs, tr, "train");
  fig.line(xs, te, "test");

  std::string out = std::string(RUN_OUTPUT_DIR) + "/depth_curve.svg";
  std::ofstream f(out);
  f << fig.render();
  f.close();
  std::cout << "figure -> " << out << "\n";
  return 0;
}