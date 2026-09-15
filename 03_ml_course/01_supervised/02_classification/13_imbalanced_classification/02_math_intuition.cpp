// Lesson 2: the weighted log-loss gradient on a single hand example.
#include "Model.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
int main() {
  std::cout << std::setprecision(4)
            << R"LESSON(Weighted log-loss: L = -(1/W) sum_i w_i [y_i log p_i + (1-y_i) log(1-p_i)],
w_i = w for positives and 1 for negatives, W = sum of weights. With
p = sigmoid(theta.x + b) the gradient is (1/W) sum_i w_i (p_i - y_i) x_i.
A positive row therefore pulls the parameters w times harder, shifting the
boundary toward the negative cloud and raising recall at the cost of more
false alarms.)LESSON"
            << "\n\n";
  // Two rows: a positive at x=1 with prediction 0.2 and a negative at x=-1
  // with prediction 0.3.  Show the gradient on theta for w = 1 and w = 8.
  for (double w : {1.0, 8.0}) {
    const double gp = w * (0.2 - 1.0) * 1.0, gn = 1.0 * (0.3 - 0.0) * -1.0;
    std::cout << "w=" << w << ": positive term " << gp << ", negative term "
              << gn << ", mean gradient " << (gp + gn) / (w + 1) << '\n';
  }
  course::WeightedLogistic m(1.0, 0.5, 500, 0.0);
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  m.fit(X, y);
  std::cout << "Fitted predictions on a separable 1-D fixture: "
            << ml::print_vec(m.predict(X)) << '\n';
}
