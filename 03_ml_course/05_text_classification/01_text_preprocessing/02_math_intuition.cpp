#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(A bag of words counts token occurrences. For "Win win now!", tokens are [win,win,now] and counts are win=2,now=1. Punctuation and order are discarded; explain the resulting loss of meaning.)LESSON"
      << "\n";
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
