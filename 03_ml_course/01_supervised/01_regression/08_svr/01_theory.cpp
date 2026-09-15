#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Epsilon SVR ignores residuals inside an epsilon-wide tube and penalizes distances outside it. C trades margin simplicity against fitting. Linear kernels give a linear function; RBF kernels compare local neighborhoods and require gamma tuning.)LESSON"
      << "\n";
  course::SVR model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {-3, -1, 3, 5};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";
}
