#include "05_text_classification/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(p(spam|x)=sigmoid(w dot x+b). Minimize mean binary cross entropy+lambda*||w||^2/2. The gradient for feature j is mean((p-y)*x_j)+lambda*w_j. The bias is unpenalized. The batch step is the reciprocal of a Hessian upper bound: 0.25*(1+max row squared norm)+lambda, including the bias coordinate.)LESSON"
      << "\n";
  text_course::Vectorizer v(true);
  v.fit({"win win now", "hello friend"});
  auto encoded = v.transform({"win unknown"});
  for (const auto &term : encoded[0])
    std::cout << v.vocabulary[term.first] << "=" << term.second << "\n";
}
