#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(idf(t)=log((1+N)/(1+df(t)))+1. For N=3,df=1, idf=1+log(2). Multiply counts by IDF then L2-normalize each document. Empty or all-unknown documents remain zero vectors.)LESSON"
      << "\n";
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
