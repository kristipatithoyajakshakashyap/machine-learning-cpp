#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(log score_c = log(prior_c)-0.5*log|Sigma_c|-0.5*(x-mu_c)^T Sigma_c^-1(x-mu_c). The class-dependent quadratic terms do not cancel. In one dimension, compare N(0,1) and N(0,4): the narrow class wins near zero and the broad class can win in both tails.)LESSON"
      << "\n";
  course::QDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
