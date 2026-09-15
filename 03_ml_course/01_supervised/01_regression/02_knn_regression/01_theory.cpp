// 01_theory.cpp
// Lesson: k-nearest-neighbours regression in one page.
// Theory summary:
//   - kNN makes NO assumption about the shape of the function; it stores the
//     training rows and predicts a new x by averaging the targets of its k
//     closest training points (distance = Euclidean by default).
//   - The ONLY hyper-parameter is k.  Small k -> high variance (wiggly fit,
//     overfits noise); large k -> high bias (smooth fit, may miss structure).
//   - This is why the data lesson sweeps k and watches train R2 collapse from
//     1 (k=1 memorises) towards a stable test R2.
// The toy below has two clusters; predicting between them shows the average
// belonging to the nearest neighbours rather than to either cluster.
// sklearn equivalent: sklearn.neighbors.KNeighborsRegressor(k).
//
// EXPECTED OUTPUT:
//   query x = 3.5
//   distances: 2.5 1.5 0.5 2.5 3.5 4.5
//   k=1  yhat 14
//   k=3  yhat 12

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "KNN.hpp"

int main() {
  std::cout << std::setprecision(7);

  ml::Mat X = {{1.0}, {2.0}, {3.0}, {6.0}, {7.0}, {8.0}};
  ml::Vec y = {10.0, 12.0, 14.0, 60.0, 62.0, 64.0};
  ml::Mat q = {{3.5}};

  std::cout << "query x = 3.5\n";
  std::cout << "distances:";
  for (const auto& row : X) {
    double d = row[0] - q[0][0];
    std::cout << " " << (d < 0 ? -d : d);
  }
  std::cout << "\n";

  for (size_t k : {1u, 3u}) {
    ml::KNNRegressor knn(k);
    knn.fit(X, y);
    std::cout << "k=" << k << "  yhat " << knn.predict(q)[0] << "\n";
  }
  return 0;
}
