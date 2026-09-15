// 03_ml_course/03_anomaly_detection/02_local_outlier_factor/01_theory.cpp
// Purpose: lesson text for Local Outlier Factor (LOF) plus a tiny 1-D example
//          with two groups of different density and one stray point.
// Inputs:  none (points are hard-coded).
// Outputs: prints only (the lesson paragraph and a point/LOF table).
// Run target: ulof_theory.
#include <iostream>

#include "LocalOutlierFactor.hpp"
int main() {
  std::cout
      << R"LESSON(Local Outlier Factor compares the density around a point with the density around its k nearest neighbours. A point whose neighbourhood is much sparser than the neighbourhoods of its neighbours receives a factor well above 1; points inside a cluster sit near 1. Unlike Isolation Forest, which asks how few random splits isolate a row, LOF asks whether the row is as crowded as the rows next to it, so it finds outliers relative to local structure, including points that a global distance rule would miss because they lie between dense and sparse clusters. Larger k smooths the estimate; k around 10-50 is the usual range. In novelty mode the training rows are stored and new rows are compared against them only.)LESSON"
      << '\n';
  // Tiny demonstration: two 1-D groups of different density and one stray.
  // k = 2 so each point is judged against its two closest neighbours.
  ml::Mat X{{0}, {0.1}, {0.2}, {0.3}, {5}, {6}, {7}, {8}, {20}};
  ml::LocalOutlierFactor m(2);
  m.fit(X);
  const auto& s = m.fitted_scores();  // LOF of each training point
  std::cout << "point lof\n";
  for (size_t i = 0; i < X.size(); ++i)
    std::cout << X[i][0] << ' ' << s[i] << '\n';
  std::cout << "The stray point at 20 has the largest factor; the sparse "
               "group scores near 1 because its neighbours are equally "
               "sparse.\n";
}
