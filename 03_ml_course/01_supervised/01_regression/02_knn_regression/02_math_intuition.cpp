// 02_math_intuition.cpp
// Lesson: distance is the whole model - the math behind kNN.
// A kNN model has nothing to learn: it stores the training rows and answers
// each query by measuring Euclidean distance to every stored point, keeping
// the k smallest, and averaging their targets.  Smaller k = tighter fit to
// the training set (low bias, high variance); larger k = smoother (high bias,
// low variance).
// sklearn equivalent: sklearn.neighbors.KNeighborsRegressor with
// metric='minkowski', p=2 (Euclidean), weights='uniform'.
//
// Worked example: query q=(3,3); neighbours A=(1,1) y=10, B=(5,2) y=20,
// C=(2,6) y=30.
//
// EXPECTED OUTPUT:
//   d(A) 2.828427
//   d(B) 2.236068
//   d(C) 3.162278
//   k=1 -> B -> 20
//   k=2 -> mean(20,10) = 15
//   k=3 -> mean(20,10,30) = 20

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  std::cout << std::setprecision(7);

  std::vector<std::vector<double>> pts = {{1, 1}, {5, 2}, {2, 6}};
  std::vector<double> y = {10, 20, 30};
  std::vector<double> q = {3, 3};

  std::vector<double> d(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    double dx = pts[i][0] - q[0], dy = pts[i][1] - q[1];
    d[i] = std::sqrt(dx * dx + dy * dy);
    std::cout << "d(" << char('A' + i) << ") " << d[i] << "\n";
  }

  // rank neighbours by distance (index permutation)
  std::vector<size_t> order(pts.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a0, size_t b0) { return d[a0] < d[b0]; });

  std::cout << "k=1 -> " << char('A' + order[0]) << " -> " << y[order[0]]
            << "\n";
  std::cout << "k=2 -> mean(" << y[order[0]] << "," << y[order[1]] << ") = "
            << (y[order[0]] + y[order[1]]) / 2.0 << "\n";
  std::cout << "k=3 -> mean(" << y[order[0]] << "," << y[order[1]] << ","
            << y[order[2]] << ") = "
            << (y[order[0]] + y[order[1]] + y[order[2]]) / 3.0 << "\n";
  return 0;
}
