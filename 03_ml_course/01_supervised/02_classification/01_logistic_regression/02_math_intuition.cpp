// 02_math_intuition.cpp
// Lesson: the sigmoid and one gradient step, by hand.
// Logistic regression squeezes the linear score z = w.x + b into (0,1):
//   p = sigmoid(z) = 1/(1+exp(-z)).
// For a single row the cross-entropy loss is L = -y*log(p) - (1-y)*log(1-p),
// and its gradient w.r.t. the parameters is simply (p - y) times the input:
//   dL/dw = (p-y)*x,  dL/db = (p-y).
// That one formula is the entire training loop.
// sklearn equivalent: sklearn.linear_model.LogisticRegression.
//
// Worked step: w=1, b=0, x=2, y=1, lr=0.1.
//
// EXPECTED OUTPUT:
//   z = 2  p = sigmoid(2) = 0.8807971
//   cross-entropy L = 0.126928
//   dL/dw = -0.2384058  dL/db = -0.1192029
//   w <- 1.023841  b <- 0.01192029

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double w = 1, b = 0, x = 2, y = 1, lr = 0.1;
  double z = w * x + b;
  double p = 1.0 / (1.0 + std::exp(-z));
  std::cout << "z = " << z << "  p = sigmoid(2) = " << p << "\n";
  std::cout << "cross-entropy L = " << -std::log(p) << "\n";
  double dw = (p - y) * x, db = p - y;
  std::cout << "dL/dw = " << dw << "  dL/db = " << db << "\n";
  std::cout << "w <- " << w - lr * dw << "  b <- " << b - lr * db << "\n";
  return 0;
}
