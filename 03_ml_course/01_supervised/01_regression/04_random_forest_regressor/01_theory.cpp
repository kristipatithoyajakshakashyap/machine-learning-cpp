// 01_theory.cpp
// Lesson: random forest regression in one page.
// Theory summary:
//   - A forest trains n_trees independent CART trees and averages their
//     predictions:  yhat = (1/T) sum_t tree_t(x).
//   - Each tree sees its own BOOTSTRAP sample (n rows drawn with
//     replacement), and each split considers only max_features random
//     features.  Both tricks DECORRELATE the trees.
//   - Averaging decorrelated learners lowers variance: Var(mean) =
//     Var(tree) * (rho + (1-rho)/T), where rho is the trees' correlation.
//     More trees rarely hurt; they only stop helping after a plateau.
// sklearn equivalent: sklearn.ensemble.RandomForestRegressor.
//
// Worked example: three shallow trees answer a query with different leaves.
//
// EXPECTED OUTPUT:
//   tree predictions: 10 14 16
//   forest prediction (mean) 13.33333
//   individual predictions spread: std 2.494438
//   correlated errors partly cancel => forest is smoother than any one tree

#include <cmath>
#include <iomanip>
#include <iostream>

#include "helper/math/metrics.hpp"

int main() {
  std::cout << std::setprecision(7);

  ml::Vec preds = {10, 14, 16};
  std::cout << "tree predictions:";
  for (double p : preds) std::cout << " " << p;
  std::cout << "\n";
  std::cout << "forest prediction (mean) " << ml::mean(preds) << "\n";
  std::cout << "individual predictions spread: std " << std::sqrt(ml::variance(preds))
            << "\n";
  std::cout
      << "correlated errors partly cancel => forest is smoother than any one tree\n";
  return 0;
}
