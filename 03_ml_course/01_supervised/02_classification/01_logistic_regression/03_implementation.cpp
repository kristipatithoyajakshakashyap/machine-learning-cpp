// 01_logistic_regression.cpp
// Lesson: binary logistic regression on breast cancer.
// Logistic regression models the log-odds of the positive class as a linear
// function:  log(p/(1-p)) = w . x + b, so p = sigmoid(w.x + b).  We train the
// parameters by minimising binary cross-entropy with Adam (plus a small L2
// term so the weights do not blow up - sklearn's C = 1).
// Equivalent: sklearn.linear_model.LogisticRegression(C=1.0, solver='lbfgs').
// The lbfgs solver converges to a marginally different optimum than our Adam
// run, so comparison is tolerance-based (~1e-4 accuracy/loss).
//
// Breast cancer: 569 rows x 30 features, 0 = benign, 1 = malignant.
// EXPECTED OUTPUT (80/20 seed 7, C=1, 1500 Adam epochs, lr 0.01):
// train accuracy 0.9868132  test 0.9649123
// test AUC 0.9858108

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../../01_regression/01_linear_regression/LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_breast_cancer(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c01_logistic.split");

  // Scale features (z-score) with training-fold statistics only.
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);
  ml::Vec mu, sd;
  ml::Mat Xtr = ml::scale_fit(tr.X, &mu, &sd);
  ml::Mat Xte = ml::scale_apply(te.X, mu, sd);

  ml::seed_rng(3);
  ml::LogisticRegression model(2, 1.0, 0.01, 1500);
  model.fit(Xtr, tr.y);

  auto tr_pred = model.predict(Xtr);
  auto te_pred = model.predict(Xte);
  std::cout << "train accuracy " << ml::accuracy(tr.y, tr_pred) << "  test "
            << ml::accuracy(te.y, te_pred) << "\n";

  // Probability of class 1 is the ROC/AUC score.
  auto pr = model.predict_proba(Xte);
  ml::Vec scores(te.y.size());
  for (size_t i = 0; i < te.y.size(); ++i) scores[i] = pr[i][1];
  std::cout << "test AUC " << ml::auc(te.y, scores) << "\n";
  return 0;
}
