#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(The primal is ||w||^2/2 + C*sum(max(0,|y-f(x)|-epsilon)). The signed dual minimizes beta^T K beta/2 - y^T beta + epsilon*sum|beta| subject to sum(beta)=0 and |beta_i|<=C. With epsilon=0.1, residual 0.05 has zero loss and residual 0.4 has loss 0.3. Prediction is sum beta_i*K(x_i,x)+b.)LESSON"
      << "\n";
  course::SVR model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
