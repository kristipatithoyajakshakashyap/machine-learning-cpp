// 03_ml_course/06_recommender_systems/03_matrix_factorization/tests/model_test.cpp
//
// Purpose : numerical fixture for the shared recommender stack (Popularity,
//           ItemCF, MatrixFactorization, recommend, ranking) on a six-rating
//           toy set. The same fixture is compiled once per recommender module
//           so each module has its own CTest entry.
// Inputs  : none (in-memory ratings only; no data files, no defines used).
// Outputs : prints nothing; exit code 0 on success, uncaught exception
//           (non-zero exit) on failure.
// Target  : rec_mf_tests (CTest: rec_mf_numerical)
#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <stdexcept>
// Minimal assertion: throwing makes the process exit non-zero, which CTest
// reports as a failure.
void check(bool ok) {
  if (!ok)
    throw std::runtime_error("Recommender fixture failed");
}
int main() {
  // Three users, three items. Rating = {user, item, id, value, timestamp}.
  // User 1 rated items 1 (5) and 2 (1); user 2 rated 1 (4) and 3 (5);
  // user 3 rated 2 (2) and 3 (4). Every item has exactly two ratings.
  std::vector<rec_course::Rating> r = {{1, 1, 1, 5, 1}, {1, 2, 2, 1, 2},
                                       {2, 1, 3, 4, 3}, {2, 3, 4, 5, 4},
                                       {3, 2, 5, 2, 5}, {3, 3, 6, 4, 6}};
  rec_course::Popularity p;
  p.fit(r);
  // Cold user and cold item must fall back to a finite estimate (global
  // mean), never index out of range or return NaN.
  check(std::isfinite(p.predict(999, 999)));
  // Recommendations never include items the user already rated in training.
  auto top = rec_course::recommend(p, 1);
  for (const auto &t : top)
    check(!p.seen(1, t.first));
  // ItemCF with k = 2 and MatrixFactorization must also handle cold pairs.
  rec_course::ItemCF c(2);
  c.fit(r);
  check(std::isfinite(c.predict(999, 999)));
  rec_course::MatrixFactorization m;
  m.fit(r);
  check(std::isfinite(m.predict(999, 999)));
  // Archive round trip reproduces the same prediction (tolerance 1e-12).
  std::stringstream s;
  m.save(s);
  rec_course::MatrixFactorization copy;
  copy.load(s);
  check(std::abs(copy.predict(1, 3) - m.predict(1, 3)) < 1e-12);
  // Loading a MatrixFactorization archive into a Popularity object must be
  // rejected by the type tag before any state is read.
  std::stringstream wrong;
  m.save(wrong);
  bool rejected = false;
  try {
    rec_course::Popularity incompatible;
    incompatible.load(wrong);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected);
  // Ranking: user 1's only unseen item is 3 and the test rates it 5, so the
  // single relevant item is at rank 1 -> Recall@10 = NDCG@10 = 1.
  std::vector<rec_course::Rating> test = {{1, 3, 7, 5, 7}};
  auto ranking = rec_course::ranking(p, test);
  check(ranking.recall == 1 && ranking.ndcg == 1);
}
