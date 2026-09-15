// 05_c_curve.cpp
// Lesson: end-to-end, how the SVM penalty C controls the margin.
// Small C tolerates training errors -> a WIDER margin, lower variance but
// more bias; large C fights every misclassification -> a NARROWER margin,
// higher variance.  On breast cancer the accuracy curve is gentle, which is
// the point: SVMs are robust to C over a wide range.
// Data: breast_cancer.csv (569 x 30).  sklearn equivalent: sklearn.svm.SVC.
//
// EXPECTED OUTPUT:
//   breast_cancer 569 x 30
//   C 0.001     test accuracy 0.8508772
//   C 1         test accuracy 0.9210526
//   C 100       test accuracy 0.9298246   <- best (margin still narrowing)
//   best test accuracy at C 100
//   figure -> results/05_c_curve_results/c_curve.svg

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"
#include "SVM.hpp"

int main() {
  std::cout << std::setprecision(7);

  auto ds = ml::load_breast_cancer(std::string(DATA_DIR));
  std::cout << "breast_cancer " << ds.n() << " x " << ds.p() << "\n";

  auto fold = ml::train_test_indices(ds.n(), 0.8, 42);
  auto train = ml::select_rows(ds, fold.train);
  auto test = ml::select_rows(ds, fold.test);

  ml::Plot fig(600, 380);
  fig.title("SVM penalty C vs accuracy (breast cancer)");
  fig.xlabel("log10(C)"); fig.ylabel("accuracy");

  const std::vector<double> Cs = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
  ml::Vec xs, te;
  double best = -1, bestC = 0;
  for (double C : Cs) {
    ml::LinearSVM svm(C, 0.01, 3000);
    svm.fit(train.X, train.y);
    double acc = ml::accuracy(test.y, svm.predict(test.X));
    xs.push_back(std::log10(C));
    te.push_back(acc);
    std::cout << "C " << std::left << std::setw(8) << C
              << "  test accuracy " << acc << "\n";
    if (acc > best) { best = acc; bestC = C; }
  }
  std::cout << "best test accuracy at C " << bestC << "\n";

  fig.line(xs, te, "test accuracy");

  std::string out = std::string(RUN_OUTPUT_DIR) + "/c_curve.svg";
  std::ofstream f(out);
  f << fig.render();
  f.close();
  std::cout << "figure -> " << out << "\n";
  return 0;
}