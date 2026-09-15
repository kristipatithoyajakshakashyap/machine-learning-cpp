#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(LDA models each class as a Gaussian with its own mean and a common covariance. Bayes rule combines likelihoods and empirical class priors. The common quadratic term cancels, leaving linear decision boundaries. Strong covariance or Gaussian violations can damage calibration.)LESSON"
      << "\n";
  course::LDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
