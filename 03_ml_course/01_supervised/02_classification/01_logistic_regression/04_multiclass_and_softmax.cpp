// 11_multiclass_strategies.cpp
// Lesson: one-vs-rest vs multinomial (softmax) logistic regression on wine.
// Logistic regression is intrinsically binary; two strategies extend it to K
// classes:
//   one-vs-rest: fit K binary classifiers (class k vs the rest) and pick the
//                one whose decision is most confident.
//   multinomial: a single softmax model with K weight vectors trained on
//                cross-entropy directly.
// Our LogisticRegression uses the multinomial path for K>2; the lesson
// compares it to hand-rolled one-vs-rest using the same optimisation, and
// prints both confusion matrices.
// sklearn equivalent: LogisticRegression(multi_class='multinomial' | 'ovr').
// Tolerance-based comparison (lbfgs vs Adam).
//
// Wine: 178 x 13, 3 classes.
// EXPECTED OUTPUT (80/20 seed 7):
// softmax   train accuracy 0.915493  test 0.8888889
// one-vs-rest test accuracy 0.8611111

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "LogisticRegression.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_wine(DATA_DIR);

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c11_multiclass.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  // Multinomial softmax.
  ml::seed_rng(17);
  ml::LogisticRegression softmax(3, 1.0, 0.01, 1500);
  softmax.fit(tr.X, tr.y);
  std::cout << "softmax   train accuracy "
            << ml::accuracy(tr.y, softmax.predict(tr.X)) << "  test "
            << ml::accuracy(te.y, softmax.predict(te.X)) << "\n";

  // One-vs-rest: one binary model per class (k vs all others); the class k
  // confidence is that binary model's P(class k) and the argmax wins.
  ml::seed_rng(17);
  const size_t K = 3;
  ml::Mat proba(te.X.size(), ml::Vec(K, 0.0));
  for (size_t k = 0; k < K; ++k) {
    ml::Vec ybin(tr.y.size());
    for (size_t i = 0; i < tr.y.size(); ++i)
      ybin[i] = (tr.y[i] == double(k)) ? 1.0 : 0.0;
    ml::LogisticRegression bin(2, 1.0, 0.01, 1500);
    bin.fit(tr.X, ybin);
    auto pr = bin.predict_proba(te.X);
    for (size_t i = 0; i < te.X.size(); ++i) proba[i][k] = pr[i][1];
  }
  ml::Vec te_pred(te.X.size(), 0.0);
  for (size_t i = 0; i < te.X.size(); ++i) {
    size_t best = 0;
    for (size_t k = 1; k < K; ++k)
      if (proba[i][k] > proba[i][best]) best = k;
    te_pred[i] = double(best);
  }
  std::cout << "one-vs-rest test accuracy " << ml::accuracy(te.y, te_pred)
            << "\n";
  return 0;
}
