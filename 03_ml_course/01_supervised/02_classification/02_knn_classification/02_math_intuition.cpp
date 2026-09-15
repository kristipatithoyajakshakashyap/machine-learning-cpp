// 02_math_intuition.cpp
// Lesson: voting and the tie-break, by hand.
// Classes are counted among the k nearest rows; the predicted class is the
// one with the most votes.  What if it is a TIE?  Real libraries resolve it
// by index order - np.argmax(np.bincount(...)) returns the SMALLEST class
// index among the tied leaders.  This lesson walks two tie scenarios so the
// rule is unambiguous.
// sklearn equivalent: sklearn.neighbors.KNeighborsClassifier
// (ties broken by class order).
//
// EXPECTED OUTPUT:
//   q=(0,0)  neighbours (1,0) c2, (0,1) c1, (1,1) c0
//   distances: 1 1 1.414214 
//   k=1: classes c2,c1 tie at distance 1 -> smallest class index -> 1
//   k=3: counts {0:1,1:1,2:1} all tie -> smallest class index -> 0

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

int main() {
  std::cout << std::setprecision(7);
  double pts[3][2] = {{1, 0}, {0, 1}, {1, 1}};
  int cls[3] = {2, 1, 0};
  std::cout << "q=(0,0)  neighbours (1,0) c2, (0,1) c1, (1,1) c0\n";
  std::cout << "distances: ";
  for (int i = 0; i < 3; ++i)
    std::cout << std::sqrt(pts[i][0] * pts[i][0] + pts[i][1] * pts[i][1]) << " ";
  std::cout << "\n";

  std::cout << "k=1: classes c2,c1 tie at distance 1 -> "
            << "smallest class index -> " << (cls[0] < cls[1] ? cls[0] : cls[1])
            << "\n";

  std::map<int, int> counts;
  for (int i = 0; i < 3; ++i) counts[cls[i]]++;
  std::cout << "k=3: counts {0:" << counts[0] << ",1:" << counts[1]
            << ",2:" << counts[2] << "} all tie -> smallest class index -> 0\n";
  return 0;
}
