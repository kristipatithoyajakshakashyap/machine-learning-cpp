// 02_math_intuition.cpp
// Lesson: one backpropagation step, by hand.
// For a single linear neuron with loss L = (yhat - y)^2:
//   dL/dw = 2*(yhat-y)*x ,  dL/db = 2*(yhat-y),  and Adam/SGD walks
//   w <- w - lr*dL/dw.  The gradient points uphill; stepping against it
//   lowers the loss.  Deeper nets repeat this exact chain rule per layer.
// sklearn equivalent: the "solver" inside
// sklearn.neural_network.MLPRegressor.
//
// Worked step: x=1, w=0.5, b=0, y=2, lr=0.1.
//
// EXPECTED OUTPUT:
//   yhat = 0.5  loss = 2.25
//   dL/dw = -3  dL/db = -3
//   w <- 0.8  b <- 0.3
//   new yhat = 1.1  new loss = 0.81

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double x = 1, w = 0.5, b = 0, y = 2, lr = 0.1;
  double yhat = w * x + b;
  double loss = (yhat - y) * (yhat - y);
  double dw = 2 * (yhat - y) * x;
  double db = 2 * (yhat - y);
  std::cout << "yhat = " << yhat << "  loss = " << loss << "\n";
  std::cout << "dL/dw = " << dw << "  dL/db = " << db << "\n";
  double wn = w - lr * dw, bn = b - lr * db;
  std::cout << "w <- " << wn << "  b <- " << bn << "\n";
  double yhat2 = wn * x + bn;
  std::cout << "new yhat = " << yhat2 << "  new loss = "
            << (yhat2 - y) * (yhat2 - y) << "\n";
  return 0;
}
