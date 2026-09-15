#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Counts represent occurrence frequency; TF-IDF discounts words common across the training corpus. Vocabulary is fitted on training documents only, sorted by document frequency with lexical tie-breaking, and capped to control memory. Unknown terms are ignored.)LESSON"
      << "\n";
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
