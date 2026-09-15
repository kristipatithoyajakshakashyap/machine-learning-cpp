#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(The model assumes class-conditional independence of token occurrences. This simplifying assumption is rarely literally true, but often yields a strong text baseline. Laplace smoothing prevents a single unseen token from assigning zero likelihood.)LESSON"
      << "\n";
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
