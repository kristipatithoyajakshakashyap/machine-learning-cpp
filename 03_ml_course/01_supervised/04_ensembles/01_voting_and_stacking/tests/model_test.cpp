// 03_ml_course/01_supervised/04_ensembles/01_voting_and_stacking/tests/model_test.cpp
//
// Purpose : numerical fixture test for course::VotingStacking in all three
//           modes. A failed check throws, so the CTest entry `ens_numerical`
//           fails on a non-zero exit code.
// Inputs  : none (a 90-row, 3-class blob fixture is generated here); no
//           defines used.
// Outputs : prints only (nothing is written to results/).
// Run     : target ens_tests; registered as `ctest -R ens_numerical`.
#include "../Model.hpp"
#include "helper/math/metrics.hpp"
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
  // Three well-separated Gaussian blobs: every member agrees (unanimous).
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 0.3);
  ml::Mat X;
  ml::Vec y;
  for (int i = 0; i < 90; ++i) {
    const int c = i % 3;
    X.push_back({c * 4.0 + noise(rng), (c == 1 ? 3.0 : 0.0) + noise(rng)});
    y.push_back(c);
  }
  // Hard voting: first confirm the fixture really is unanimous (all three
  // members give the forest's labels), then the majority vote must equal it.
  ml::seed_rng(3);
  course::VotingStacking hard(course::VotingStacking::kHard);
  hard.fit(X, y);
  const ml::Vec base = hard.forest().predict(X);
  check(hard.logistic().predict(X) == base && hard.knn().predict(X) == base,
        "fixture is unanimous");
  check(hard.predict(X) == base, "hard voting equals the unanimous member");

  // Soft voting: the mean of three probability rows is itself a probability
  // row, so each output row must sum to one (1e-9 absorbs rounding).
  ml::seed_rng(3);
  course::VotingStacking soft(course::VotingStacking::kSoft);
  soft.fit(X, y);
  for (const auto &row : soft.predict_proba(X)) {
    double s = 0;
    for (double p : row)
      s += p;
    check(std::abs(s - 1.0) < 1e-9, "soft probabilities sum to one");
  }

  // Stacking: one prediction per row, > 95 % accuracy on separable blobs, and
  // the meta-learner sees 3 members x 3 classes = 9 stacked features.
  ml::seed_rng(3);
  course::VotingStacking stack(course::VotingStacking::kStacking);
  stack.fit(X, y);
  const ml::Vec pred = stack.predict(X);
  check(pred.size() == y.size(), "stacking predicts every row");
  check(ml::accuracy(y, pred) > 0.95, "stacking fits separable blobs");
  check(stack.meta().n_features() == 9, "meta input is 3 members x 3 classes");

  // Persistence for every mode: the copy is constructed in hard mode, so the
  // archive must restore the mode and every member; probabilities are
  // compared bit for bit because save/load is lossless.
  for (const course::VotingStacking *m : {&hard, &soft, &stack}) {
    std::stringstream stream;
    m->save(stream);
    course::VotingStacking copy(0);
    copy.load(stream);
    const ml::Mat a = m->predict_proba(X), b = copy.predict_proba(X);
    for (size_t i = 0; i < a.size(); ++i)
      for (size_t c = 0; c < a[i].size(); ++c)
        check(a[i][c] == b[i][c], "save/load reproduces probabilities exactly");
    check(copy.mode() == m->mode(), "mode survives reload");
  }

  // Input validation: mode 3 does not exist, and empty data must be rejected;
  // both raise std::invalid_argument.
  bool rejected = false;
  try {
    (void)course::VotingStacking(3);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "invalid mode throws");
  rejected = false;
  try {
    hard.fit({}, {});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "empty data throws");
}
