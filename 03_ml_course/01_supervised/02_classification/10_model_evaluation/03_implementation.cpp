// 13_classifier_shootout.cpp
// Lesson: every classifier from this track head-to-head on titanic, scored by
// stratified 5-fold cross-validation.  Each model is re-fit per fold and the
// reported number is the mean test-set accuracy across folds.  This is the
// "model selection" moment: no single algorithm wins everywhere, and the table
// shows the trade-off between simple interpretable models (logistic, naive
// Bayes) and flexible ones (GBDT, forest, MLP).
//
// Every fold is exported via write_split so the same rows can be independently
// audited. Median imputation and scaling are fitted within each training fold.
//
// EXPECTED OUTPUT (titanic, stratified 5-fold, seed 13):
// logistic  acc 0.805791
// knn5      acc 0.693631
// tree      acc 0.8058351
// nb        acc 0.7654038
// svm       acc 0.6868707
// forest    acc 0.8226704
// gbdt      acc 0.8338878
// adaboost  acc 0.8069085
// mlp       acc 0.7821826

#include <iomanip>
#include <iostream>
#include <functional>
#include <vector>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "../../01_regression/04_random_forest_regressor/RandomForest.hpp"
#include "../../01_regression/02_knn_regression/KNN.hpp"
#include "../../01_regression/01_linear_regression/LinearRegression.hpp"
#include "../01_logistic_regression/LogisticRegression.hpp"
#include "helper/math/metrics.hpp"
#include "../04_naive_bayes/GaussianNB.hpp"
#include "../09_mlp_classifier/MLP.hpp"
#include "helper/math/optim.hpp"
#include "helper/preprocessing/preprocessor.hpp"
#include "../03_svm_classification/SVM.hpp"

namespace {

// Generic adapter: given a callable of the form (Xtr, ytr, Xte) -> yhat_te,
// run 5-fold stratified CV and average the test-set accuracy.
double cv_accuracy(const ml::Dataset& d,
                   const std::vector<ml::Fold>& folds,
                   const std::function<ml::Vec(const ml::Mat&, const ml::Vec&,
                                               const ml::Mat&)>& fit) {
  double total = 0.0;
  for (const auto& f : folds) {
    auto tr = ml::select_rows(d, f.train);
    auto te = ml::select_rows(d, f.test);
    ml::Preprocessor prep;
    prep.fit(tr.X);
    ml::Vec pred = fit(prep.transform(tr.X), tr.y, prep.transform(te.X));
    total += ml::accuracy(te.y, pred);
  }
  return total / static_cast<double>(folds.size());
}

}  // namespace

int main() {
  std::cout << std::setprecision(7);
  auto data = ml::load_titanic(DATA_DIR,true);
  auto folds = ml::stratified_kfold(data.y, 5, 13);
  std::string sp = std::string(RUN_OUTPUT_DIR) + "/c13_shootout";
  for (size_t i = 0; i < folds.size(); ++i)
    ml::write_split(folds[i], sp + std::to_string(i) + ".split");

  // Logistic regression (multinomial).
  auto logistic = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::seed_rng(3);
    ml::LogisticRegression m(2, 1.0, 0.01, 800);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "logistic  acc "
            << cv_accuracy(data, folds, logistic) << "\n";

  // k-NN k=5.
  auto knn = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::KNNClassifier m(5);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "knn5      acc "
            << cv_accuracy(data, folds, knn) << "\n";

  // Decision tree classifier, depth 5.
  auto tree = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::DecisionTree m(ml::TreeTask::Classification, ml::SplitCriterion::Gini,
                       5, 3);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "tree      acc "
            << cv_accuracy(data, folds, tree) << "\n";

  // Gaussian naive Bayes.
  auto nb = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::GaussianNB m;
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "nb        acc "
            << cv_accuracy(data, folds, nb) << "\n";

  // Linear SVM.
  auto svm = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::seed_rng(3);
    ml::LinearSVM m(1.0, 0.01, 800);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "svm       acc "
            << cv_accuracy(data, folds, svm) << "\n";

  // Random forest, 100 trees.
  auto forest = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::seed_rng(5);
    ml::RandomForest m(100, 0, 10, 1);
    m.set_task(ml::TreeTask::Classification);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "forest    acc "
            << cv_accuracy(data, folds, forest) << "\n";

  // Gradient boosted trees, 100 rounds.
  auto gbdt = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::GradientBoostedTrees m(100, 0.1, 2, 1, 1.0);
    m.set_task(ml::TreeTask::Classification);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "gbdt      acc "
            << cv_accuracy(data, folds, gbdt) << "\n";

  // AdaBoost, 100 stumps.
  auto adaboost = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::seed_rng(11);
    ml::AdaBoost m(100, 1.0);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "adaboost  acc "
            << cv_accuracy(data, folds, adaboost) << "\n";

  // MLP, one hidden layer of 16.
  auto mlp = [&](const ml::Mat& Xtr, const ml::Vec& ytr, const ml::Mat& Xte) {
    ml::seed_rng(7);
    ml::MLP m({16}, 2, 0.01, 150);
    m.fit(Xtr, ytr);
    return m.predict(Xte);
  };
  std::cout << "mlp       acc "
            << cv_accuracy(data, folds, mlp) << "\n";

  return 0;
}
