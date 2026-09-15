// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/01_theory.cpp
//
// Purpose : Lesson 1 - what permutation importance and learning curves measure.
//           Prints the lesson text, then fits a 20-tree forest on a 6-row,
//           3-class fixture to show the shared model factory in action.
// Inputs  : none (fixture is inline); no defines used.
// Outputs : prints only.
// Run     : target diag_theory (no arguments).
//
// Lesson 1: what permutation importance and learning curves measure.
#include "Forest.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Permutation importance asks: how much does a fitted model's score fall
when one feature column is shuffled so it no longer carries information?
It is model-agnostic, measured on held-out rows, and honest about
correlated features (each shuffled column can be compensated by its twin,
so both look unimportant). Impurity importance, by contrast, is a
training-time byproduct of trees that favours high-cardinality features.

A learning curve fits the model on growing fractions of the training data
and records the training score and the cross-validated score. A wide gap
that narrows slowly means variance: more data helps. Two low curves that
have converged mean bias: the model family is too weak. Neither can be
diagnosed from a single fit.)LESSON"
      << "\n";
  // Feature 0 alone identifies the class (0, 1, 2); feature 1 is a distractor.
  auto f = course::make_forest(20);
  ml::Mat X = {{0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 1}, {2, 2}};
  ml::Vec y = {0, 0, 1, 1, 2, 2};
  ml::seed_rng(1); // bootstrap draws come from the shared course RNG
  f.fit(X, y);
  std::cout << "Forest of " << f.n_trees() << " trees on a 3-class fixture: "
            << ml::print_vec(f.predict(X)) << "\n";
}
