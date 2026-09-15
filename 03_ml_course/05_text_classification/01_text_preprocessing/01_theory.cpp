#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Tokenization converts variable strings to sequences of words. This course lowercases ASCII letters and splits on non-alphanumeric characters. Preserve original row IDs and remove exact duplicate messages before splitting. The final lesson shows how these choices affect a spam classifier.)LESSON"
      << "\n";
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
