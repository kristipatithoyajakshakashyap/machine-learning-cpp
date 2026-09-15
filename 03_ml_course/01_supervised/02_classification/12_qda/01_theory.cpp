#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(QDA allows a different Gaussian covariance for every class. Boundaries can curve, at the cost of estimating many more parameters. Small classes and redundant features make covariance estimates unstable; diagonal regularization improves numerical conditioning.)LESSON"
      << "\n";
  course::QDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
