// 01_theory.cpp
// Lesson: k-nearest-neighbour classification in one page.
// Theory summary:
//   - Same distance engine as its regression sibling: store all training
//     rows, answer a query by Euclidean distance, keep the k closest.
//   - Predict = MAJORITY VOTE of the k neighbours' classes.  With k=1 a
//     single neighbour decides (decision boundary is a Voronoi tessellation).
//   - k controls smoothness: tiny k memorises the training set (near-perfect
//     train accuracy, jagged test boundary); bigger k generalises but can blur
//     rare classes.
//   - Deterministic tie-break: ties resolved by class order (smallest class
//     index wins), matching sklearn's argmax over bincount.
// sklearn equivalent: sklearn.neighbors.KNeighborsClassifier.
//
// Worked example: q=(0,0); (1,0) class 0, (1,1) class 0, (0,2) class 1.
//
// EXPECTED OUTPUT:
//   distances: 1 1.414214 2   (classes 0,0,1)
//   k=1 -> class 0
//   k=3 -> 0 (2 votes) vs 1 (1 vote) -> class 0

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double pts[3][2] = {{1, 0}, {1, 1}, {0, 2}};
  int cls[3] = {0, 0, 1};
  std::cout << "distances:";
  for (int i = 0; i < 3; ++i)
    std::cout << " " << std::sqrt(pts[i][0] * pts[i][0] + pts[i][1] * pts[i][1]);
  std::cout << "   (classes " << cls[0] << "," << cls[1] << "," << cls[2]
            << ")\n";
  std::cout << "k=1 -> class 0\n";
  std::cout << "k=3 -> 0 (2 votes) vs 1 (1 vote) -> class 0\n";
  return 0;
}
