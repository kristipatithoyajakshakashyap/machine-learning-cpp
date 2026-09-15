// 03_ml_course/03_anomaly_detection/02_local_outlier_factor/02_math_intuition.cpp
// Purpose: the LOF definitions (k-distance, reachability distance, local
//          reachability density, factor) worked by hand on four 1-D points and
//          checked numerically against the implementation.
// Inputs:  none (points A=0, B=1, C=2, D=10 are hard-coded, k = 2).
// Outputs: prints only; throws (non-zero exit) if any factor deviates from the
//          hand result by more than 1e-12.
// Run target: ulof_math_intuition.
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "LocalOutlierFactor.hpp"
int main() {
  std::cout
      << R"LESSON(Definitions with k nearest neighbours N_k(p):
  k-distance(p)      = distance to the k-th nearest neighbour
  reach_k(p, o)      = max(k-distance(o), d(p, o))
  lrd_k(p)           = k / sum_{o in N_k(p)} reach_k(p, o)
  LOF_k(p)           = mean_{o in N_k(p)} lrd_k(o) / lrd_k(p)
Hand example, 1-D points A=0 B=1 C=2 D=10, k=2:
  k-distances: A 2, B 1, C 2, D 9
  lrd: A 2/3, B 1/2, C 2/3, D 2/17
  LOF: A 7/8, B 4/3, C 7/8, D 119/24 = 4.958)LESSON"
      << '\n';
  // Same four points as the hand example; the exact fractions above are the
  // reference values.
  ml::Mat X{{0}, {1}, {2}, {10}};
  ml::LocalOutlierFactor m(2);
  m.fit(X);
  const auto& lof = m.fitted_scores();
  const double expected[] = {7.0 / 8, 4.0 / 3, 7.0 / 8, 119.0 / 24};
  const char* names = "ABCD";
  for (size_t i = 0; i < 4; ++i) {
    std::cout << names[i] << " computed " << lof[i] << " expected "
              << expected[i] << '\n';
    // Exact rational arithmetic in the hand example -> tight tolerance.
    if (std::abs(lof[i] - expected[i]) > 1e-12)
      throw std::runtime_error("hand computation mismatch");
  }
  std::cout << "Numerical check passed: implementation matches the hand "
               "computation.\n";
}
