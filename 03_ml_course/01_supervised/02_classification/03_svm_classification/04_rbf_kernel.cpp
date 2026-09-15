// 06_svm_rbf_kernel.cpp
// Lesson: RBF kernel SVM on a two-feature slice of the iris (petal dimensions)
// so the non-linear decision boundary can be visualised in 2-D.
// A linear SVM can only draw straight boundaries; the RBF kernel
//   K(x,x') = exp(-gamma ||x-x'||^2)
// maps the data into an infinite-feature space where a simple hyperplane
// becomes a non-linear surface in the original space.  gamma controls the
// influence radius of a single point (too large -> overfit), C trades margin
// size against training-error tolerance.
//
// We solve the SVM dual with Platt's SMO: keep a working alpha pair, fix the
// rest, and update until all KKT conditions are satisfied within tol.
// Equivalent: sklearn.svm.SVC(kernel='rbf', C=..., gamma=...) - libsvm's
// working-set choice differs, so comparison is tolerance-based (~1e-4 acc).
//
// EXPECTED OUTPUT (iris petal slice, 80/20 seed 7, C=10):
// rbf svm  train accuracy 1  test accuracy 1

#include <iomanip>
#include <iostream>

#include "helper/eval/cross_validation.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "SVM.hpp"

int main() {
  std::cout << std::setprecision(7);
  auto full = ml::load_iris(DATA_DIR);

  // Reduce to two features: petal_length (col 2) and petal_width (col 3).
  ml::Dataset data;
  data.feature_names = {"petal_length", "petal_width"};
  data.target_name = "species";
  data.y = full.y;
  for (size_t i = 0; i < full.n(); ++i)
    data.X.push_back({full.X[i][2], full.X[i][3]});

  // Binary task: species 0 (setosa) vs the rest (squeeze labels to 0/1).
  for (double& v : data.y) v = (v == 0.0) ? 1.0 : 0.0;

  auto fold = ml::train_test_indices(data.n(), 0.8, 7);
  ml::write_split(fold, std::string(RUN_OUTPUT_DIR) + "/c06_svm.split");
  auto tr = ml::select_rows(data, fold.train);
  auto te = ml::select_rows(data, fold.test);

  ml::KernelSVM model(10.0, -1.0, 1e-3, 100);  // gamma -1 => auto (1/p)
  model.fit(tr.X, tr.y);
  std::cout << "rbf svm  train accuracy "
            << ml::accuracy(tr.y, model.predict(tr.X)) << "  test accuracy "
            << ml::accuracy(te.y, model.predict(te.X)) << "\n";
  return 0;
}
