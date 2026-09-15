#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(P(t|c)=(count(t,c)+alpha)/(total_tokens(c)+alpha*V). log score(c)=log prior(c)+sum_t count(t)*log P(t|c). For class counts [3,1] and alpha=1, token probabilities are [4/6,2/6]. Normalize log scores stably to obtain probabilities.)LESSON"
      << "\n";
  text_course::Vectorizer v(false);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
