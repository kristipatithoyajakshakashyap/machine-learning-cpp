// 03_ml_course/03_anomaly_detection/03_one_class_svm/01_theory.cpp
// Purpose: lesson text for the one-class SVM (nu budget, gamma kernel width,
//          origin-separating hyperplane) plus a tiny 1-D cluster-and-stray
//          example.
// Inputs:  none (points are hard-coded; nu = 0.2, gamma = 1).
// Outputs: prints only (the lesson paragraph and a point/score table).
// Run target: uocsvm_theory.
#include <iostream>

#include "OneClassSVM.hpp"
int main() {
  std::cout
      << R"LESSON(A one-class SVM learns the region of feature space where ordinary rows live, without any labels. With an RBF kernel every row is mapped to a point on a sphere in a high-dimensional space, and the algorithm finds the hyperplane that separates those points from the origin with the largest margin. Rows that fall on the origin side are anomalies. The parameter nu in (0, 1] is a budget: at most a fraction nu of training rows lie outside the boundary and at least nu of them become support vectors, so nu 0.05 means "assume about 5 percent of my training data is unusual". gamma sets the kernel width: a small gamma gives one smooth blob, a large gamma wraps a tight boundary around each training row and treats everything else as anomalous. Unlike Isolation Forest (random splits) and LOF (local density ratios), the one-class SVM produces a smooth, deterministic boundary and stores only its support vectors.)LESSON"
      << '\n';
  // Tiny demonstration: a 1-D cluster and one stray point.
  // Score = rho - decision value, so positive means outside the boundary.
  ml::Mat X{{0}, {0.2}, {0.4}, {0.6}, {0.8}, {1.0}, {1.2}, {1.4}, {1.6}, {6}};
  ml::OneClassSVM m(.2, 1);
  m.fit(X);
  const auto s = m.score_samples(X);
  std::cout << "point score\n";
  for (size_t i = 0; i < X.size(); ++i)
    std::cout << X[i][0] << ' ' << s[i] << '\n';
  std::cout << "The stray point at 6 has the largest score (positive means "
               "outside the boundary); the cluster scores at or below "
               "zero.\n";
}
