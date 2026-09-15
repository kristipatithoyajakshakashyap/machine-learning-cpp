#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(L2-regularized logistic regression learns a weighted combination of sparse TF-IDF features. Unlike Naive Bayes it models the conditional label distribution directly. Strong shrinkage can underfit rare spam indicators.)LESSON"
      << "\n";
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
