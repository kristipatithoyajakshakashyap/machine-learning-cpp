#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(log score_c = log(prior_c)-0.5*log|Sigma|-0.5*(x-mu_c)^T Sigma^-1(x-mu_c). With equal priors, shared unit variance, and means 0 and 2, the boundary is x=1. A pooled maximum-likelihood covariance uses 1/n times summed within-class cross-products. Add lambda I for invertibility.)LESSON"
      << "\n";
  course::LDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
