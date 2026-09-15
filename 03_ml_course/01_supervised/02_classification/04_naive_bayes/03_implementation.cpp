// 04_gaussian_naive_bayes.cpp
// Lesson: Gaussian Naive Bayes on iris (3 classes) and wine (3 classes).
// Bayes' rule + the "naive" independence assumption make the posterior easy:
//   p(k|x) ~ pi_k * prod_j N(x_j | mu_kj, sigma_kj^2)
// We estimate pi_k (class frequency), mu_kj and sigma_kj^2 per class, then
// score each test row under every class's Gaussian and pick the argmax of the
// log-posterior.  Cheap, stable, and a surprisingly strong linear baseline -
// even though the "independence" assumption is false, the model often works.
// var_smoothing = 1e-9 matches sklearn's default.
// Equivalent: sklearn.naive_bayes.GaussianNB(var_smoothing=1e-9).
// Exact-match verification via the split sidecar.
//
// EXPECTED OUTPUT (80/20 seed 7):
// iris  train accuracy 0.9666667  test accuracy 0.9333333
// wine  train accuracy 0.9859155  test accuracy 0.9444444

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "GaussianNB.hpp"

namespace {
void run(const ml::Dataset& data, const std::string& name,
         const std::string& split_path) {
  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, split_path);
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);
  ml::GaussianNB model;
  model.fit(tr.X, tr.y);
  std::cout << name << "  train accuracy " << ml::accuracy(tr.y, model.predict(tr.X))
            << "  test accuracy " << ml::accuracy(te.y, model.predict(te.X)) << "\n";
}
}  // namespace

int main() {
  std::cout << std::setprecision(7);
  run(ml::load_iris(DATA_DIR), "iris", std::string(RUN_OUTPUT_DIR) + "/c04_nb_iris.split");
  run(ml::load_wine(DATA_DIR), "wine", std::string(RUN_OUTPUT_DIR) + "/c04_nb_wine.split");
  return 0;
}
