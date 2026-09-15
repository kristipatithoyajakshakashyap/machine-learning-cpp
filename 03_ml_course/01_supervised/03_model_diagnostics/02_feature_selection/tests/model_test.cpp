// 03_ml_course/01_supervised/03_model_diagnostics/02_feature_selection/tests/model_test.cpp
//
// Purpose : numerical fixture test for the selection helpers (filter_scores,
//           top_k, forward_selection, l1_path, l1_scores) and for
//           course::SelectedForest. A failed check throws, so the CTest entry
//           `fsel_numerical` fails on a non-zero exit code.
// Inputs  : none (a 120-row, 3-column fixture is generated here); no defines.
// Outputs : prints only (nothing is written to results/).
// Run     : target fsel_tests; registered as `ctest -R fsel_numerical`.
#include "../Model.hpp"
#include "helper/eval/feature_selection.hpp"
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
// Assertion helper: the message names the failing property in the exception.
void check(bool x, const char *what) {
  if (!x)
    throw std::runtime_error(std::string("Numerical fixture failed: ") + what);
}
int main() {
  // Column 0 is pure signal, column 1 is mild signal, column 2 is noise.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 1.0);
  ml::Mat X;
  ml::Vec y;
  for (int i = 0; i < 120; ++i) {
    const int c = i % 2;
    X.push_back({c * 6.0 + noise(rng) * 0.3, c * 1.0 + noise(rng), noise(rng)});
    y.push_back(c);
  }
  // Filter: one ANOVA F per column, strictly ordered signal > mild > noise,
  // and top_k(., 1) must therefore return column 0.
  const ml::Vec f = ml::filter_scores(X, y, true);
  check(f.size() == 3, "one score per feature");
  check(f[0] > f[1] && f[1] > f[2], "signal first, noise last");
  check(ml::top_k(f, 1)[0] == 0, "top_k picks the signal column");

  // Wrapper: two greedy steps with 3-fold accuracy of a 10-tree forest. The
  // first step must pick column 0 and already score above 95 % on its own.
  auto steps = ml::forward_selection([] { return course::make_forest(10); }, X,
                                     y, 2, 3, 1, ml::accuracy, true);
  check(steps.size() == 2, "two forward steps");
  check(steps[0].feature == 0, "forward selection picks the signal first");
  check(steps[0].cv_score > 0.95, "signal column alone is enough");

  // Embedded: path shape is lambdas x features, and the column that stays
  // non-zero at the strongest penalty (column 0) gets the top l1 score.
  const ml::Vec lambdas = {0.5, 0.1, 0.01};
  const ml::Mat path = ml::l1_path(X, y, lambdas);
  check(path.size() == 3 && path[0].size() == 3, "path is lambdas x features");
  const ml::Vec l1 = ml::l1_scores(path, lambdas);
  check(ml::top_k(l1, 1)[0] == 0, "lasso keeps the signal column longest");

  // SelectedForest with k = 2: kept() lists the two best columns, best first,
  // and the forest on those columns must fit the separable fixture.
  ml::seed_rng(5);
  course::SelectedForest model(2);
  model.fit(X, y);
  check(model.kept().size() == 2 && model.kept()[0] == 0, "keeps top 2");
  check(ml::accuracy(y, model.predict(X)) > 0.95, "selected forest fits");
  // Persistence: the copy is built with k = 1 so the archive must overwrite
  // k and the kept set; forest probabilities must match bit for bit.
  std::stringstream stream;
  model.save(stream);
  course::SelectedForest copy(1);
  copy.load(stream);
  check(copy.k() == 2 && copy.kept() == model.kept(), "kept set survives");
  const ml::Mat a = model.predict_proba(X), b = copy.predict_proba(X);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t c = 0; c < a[i].size(); ++c)
      check(a[i][c] == b[i][c], "save/load reproduces probabilities exactly");

  // Input validation: k = 7 on 3 columns and empty data both raise
  // std::invalid_argument instead of indexing out of range.
  bool rejected = false;
  try {
    course::SelectedForest big(7);
    big.fit(X, y);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "k larger than feature count throws");
  rejected = false;
  try {
    ml::filter_scores({}, {}, true);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "empty data throws");
}
