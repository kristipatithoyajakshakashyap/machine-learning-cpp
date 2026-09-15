#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Elastic Net combines squared-error regression with L1 sparsity and L2 shrinkage. Standardize features before comparing their penalties. Correlated predictors can share weight; coefficients describe association, not causation.)LESSON"
      << "\n";
  course::ElasticNet model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
