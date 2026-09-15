// 03_ml_course/05_text_classification/01_text_preprocessing/tests/model_test.cpp
//
// Purpose : numerical fixture for the shared text stack (Vectorizer,
//           MultinomialNB, LinearTextClassifier) on a two-document corpus.
//           The same fixture is compiled once per text module so each module
//           has its own CTest entry.
// Inputs  : none (in-memory strings only; no data files, no defines used).
// Outputs : prints only; exit code 0 on success, uncaught exception otherwise.
// Target  : text_preprocessing_tests (CTest: text_preprocessing_numerical)
#include "05_text_classification/Workflow.hpp"
#include <sstream>
#include <stdexcept>
// Minimal assertion: throwing makes the process exit non-zero, which CTest
// reports as a failure.
void check(bool x) {
  if (!x)
    throw std::runtime_error("Text fixture failed");
}
int main() {
  // Raw-count vectorizer fitted on two documents: vocabulary is
  // {friend, hello, now, win} (order by document frequency, then alphabet).
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto x = v.transform({"win win unknown", "unseen"});
  // "win" appears twice -> one non-zero feature with count 2; "unknown" was
  // never seen at fit time, so it contributes no feature at all.
  check(x[0].size() == 1 && x[0][0].second == 2);
  // A document made only of unknown tokens becomes an empty sparse row.
  check(x[1].empty());
  // fit() must not have leaked transform-time tokens into the index.
  check(v.index.find("unknown") == v.index.end());
  // Round trip through the text archive format reproduces identical rows.
  std::stringstream ss;
  v.save(ss);
  text_course::Vectorizer restored;
  restored.load(ss);
  check(restored.transform({"win win unknown", "unseen"}) == x);
  // Naive Bayes: document 0 is ham (0), document 1 is spam (1). Each model
  // must assign p(spam) < 0.5 to the ham row and > 0.5 to the spam row.
  text_course::MultinomialNB nb;
  auto train = v.transform({"hello friend", "win win now"});
  nb.fit(train, {0, 1}, v.vocabulary.size());
  auto a = nb.probabilities(train);
  check(a[0] < 0.5 && a[1] > 0.5);
  // Logistic regression on the same rows separates the two documents too.
  text_course::LinearTextClassifier lr;
  lr.fit(train, {0, 1}, v.vocabulary.size());
  auto b = lr.probabilities(train);
  check(b[0] < 0.5 && b[1] > 0.5);
}
