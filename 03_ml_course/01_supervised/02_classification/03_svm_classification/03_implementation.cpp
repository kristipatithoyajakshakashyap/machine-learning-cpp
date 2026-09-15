// 05_svm_linear.cpp
// Lesson: linear Support Vector Machine on breast cancer.
// A linear SVM finds the hyperplane w.x + b = 0 that separates the classes
// with the largest margin.  The primal objective is the hinge loss plus an L2
// regulariser:  L(w,b) = (1/n) sum max(0, 1 - y (w.x + b)) + (1/2C)||w||^2.
// The decision value w.x + b (not a probability) drives the prediction:
// sign > 0 -> class 1.
// Equivalent: sklearn.svm.SVC(kernel='linear', C=...) / SGDClassifier(loss=
// 'hinge').  Comparison is tolerance-based (~1e-4 accuracy).
//
// Breast cancer, features standardised so the margin geometry is well scaled.
// EXPECTED OUTPUT (80/20 seed 7, C=1, Adam lr 0.01, 2000 epochs):
// linear svm  train accuracy 0.9516484  test accuracy 0.9385965

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../../01_regression/01_linear_regression/LinearRegression.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"
#include "SVM.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_breast_cancer(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c05_svm.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::Vec mu, sd;
  ml::Mat Xtr = ml::scale_fit(tr.X, &mu, &sd);
  ml::Mat Xte = ml::scale_apply(te.X, mu, sd);

  ml::seed_rng(3);
  ml::LinearSVM model(1.0, 0.01, 2000);
  model.fit(Xtr, tr.y);
  std::cout << "linear svm  train accuracy "
            << ml::accuracy(tr.y, model.predict(Xtr)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(Xte)) << "\n";
  return 0;
}
