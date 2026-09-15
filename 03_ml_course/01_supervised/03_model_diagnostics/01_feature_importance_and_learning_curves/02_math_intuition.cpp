// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/02_math_intuition.cpp
//
// Purpose : Lesson 2 - the arithmetic behind both diagnostics. Prints the
//           importance and learning-curve formulas, then computes permutation
//           importance on a 30-row fixture with one informative and one noise
//           column so the expected ordering can be checked by eye.
// Inputs  : none (fixture is inline); no defines used.
// Outputs : prints only.
// Run     : target diag_math_intuition (no arguments).
//
// Lesson 2: the arithmetic behind both diagnostics on a tiny example.
#include "Forest.hpp"
#include "helper/eval/diagnostics.hpp"
#include <iomanip>
#include <iostream>
int main() {
  std::cout << std::setprecision(3)
            << R"LESSON(Importance of feature j: I_j = s(D) - mean_r s(D with column j permuted),
averaged over r repeats. s is any higher-is-better metric. A feature the
model never consults gives I_j ~ 0 with small spread; a feature the model
relies on gives a large positive drop. Negative values arise from noise and
mean "no evidence of importance".

Learning curve point at fraction f: fit on the first f*|train| rows of each
fold (per class when stratified), score on those rows (train) and on the
fold's test rows (cv), then average over folds. train - cv is the
generalisation gap; cv at f = 1 is the usual k-fold estimate.)LESSON"
            << "\n\n";
  // Feature 0 decides the class, feature 1 is noise.
  // Column 0 is i mod 3 (== the label); column 1 is (7i mod 5), unrelated to y.
  ml::Mat X;
  ml::Vec y;
  for (int i = 0; i < 30; ++i) {
    X.push_back({static_cast<double>(i % 3), static_cast<double>((i * 7) % 5)});
    y.push_back(i % 3);
  }
  ml::seed_rng(3); // forest bootstrap seed
  auto f = course::make_forest(20);
  f.fit(X, y);
  // 5 shuffle repeats per column, permutation seed 11; accuracy is the metric.
  auto imp = ml::permutation_importance(f, X, y, ml::accuracy, 5, 11);
  std::cout << "feature 0 drop " << imp[0].mean_drop << " +- " << imp[0].std_drop
            << "; feature 1 drop " << imp[1].mean_drop << " +- "
            << imp[1].std_drop << '\n';
}
