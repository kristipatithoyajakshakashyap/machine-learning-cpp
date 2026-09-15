#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(J = ||y-Xw-b||^2/(2n) + alpha*r*sum|w_j| + alpha*(1-r)*sum(w_j^2)/2. For a coordinate, let rho = mean(x_j*(residual+x_j*w_j)); update w_j = soft(rho,alpha*r)/(mean(x_j^2)+alpha*(1-r)). With rho=3, norm=2, alpha=1,r=0.5, the new coefficient is (3-0.5)/(2+0.5)=1. The intercept is unpenalized.)LESSON"
      << "\n";
  course::ElasticNet model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
