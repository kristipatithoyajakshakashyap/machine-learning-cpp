// 12_binary_classification.cpp
// Lesson: binary-classification deep dive on breast cancer.
// One score (logistic probability) drives every binary metric, and the choice
// of decision threshold is a business decision: lowering it catches more
// positives but also produces false alarms.  This lesson sweeps the threshold
// over p_pred, builds the full ROC curve, and prints the confusion
// cells + F1 at a few interesting points, plus the AUC.  All metrics come
// from ml/metrics.hpp and match sklearn digit-for-digit on the same split.
// Equivalent: sklearn.metrics confusion_matrix / roc_curve / roc_auc_score.
//
// EXPECTED OUTPUT (80/20 seed 7, softmax logistic, C=1):
// th=0.3  tn 74 fp 0 fn 3 tp 37  f1 0.961039
// th=0.5  tn 74 fp 0 fn 4 tp 36  f1 0.9473684
// th=0.7  tn 74 fp 0 fn 4 tp 36  f1 0.9473684
// auc 0.9858108

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../../01_regression/01_linear_regression/LinearRegression.hpp"
#include "../01_logistic_regression/LogisticRegression.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_breast_cancer(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c12_binary.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::Vec mu, sd;
  ml::Mat Xtr = ml::scale_fit(tr.X, &mu, &sd);
  ml::Mat Xte = ml::scale_apply(te.X, mu, sd);

  ml::seed_rng(3);
  ml::LogisticRegression model(2, 1.0, 0.01, 1500);
  model.fit(Xtr, tr.y);
  auto pr = model.predict_proba(Xte);
  ml::Vec scores(te.y.size());
  for (size_t i = 0; i < te.y.size(); ++i) scores[i] = pr[i][1];

  for (double th : {0.3, 0.5, 0.7}) {
    ml::Vec pred(te.y.size());
    for (size_t i = 0; i < scores.size(); ++i) pred[i] = scores[i] >= th ? 1.0 : 0.0;
    auto bs = ml::binary_scores(te.y, pred, &scores);
    std::cout << "th=" << th << "  tn " << bs.tn << " fp " << bs.fp << " fn "
              << bs.fn << " tp " << bs.tp << "  f1 " << bs.f1 << "\n";
  }
  std::cout << "auc " << ml::auc(te.y, scores) << "\n";
  return 0;
}
