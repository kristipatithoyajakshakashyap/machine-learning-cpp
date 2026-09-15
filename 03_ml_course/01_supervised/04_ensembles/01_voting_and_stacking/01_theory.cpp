// 03_ml_course/01_supervised/04_ensembles/01_voting_and_stacking/01_theory.cpp
//
// Purpose : Lesson 1 - what voting and stacking are and when an ensemble
//           helps. Prints the lesson text, then fits a soft-voting ensemble
//           on a 10-row toy set to show the class in action.
// Inputs  : none (fixture is inline); no defines used.
// Outputs : prints only.
// Run     : target ens_theory (no arguments).
#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(An ensemble combines several fitted models. Hard voting counts the labels the members predict; soft voting averages their class probabilities; stacking learns how much to trust each member by fitting a second-level model on the members' out-of-fold predictions. Ensembles help when the members are individually decent and make different mistakes. Read theory.md for why out-of-fold predictions are mandatory in stacking.)LESSON"
      << "\n";
  // Two classes separated by the sign of x1; x2 is an uninformative 0/1 bit.
  ml::Mat X = {{-2, 0}, {-1, 1}, {1, 0}, {2, 1}, {-3, 1},
               {3, 0}, {-1.5, 0}, {1.5, 1}, {-2.5, 1}, {2.5, 0}};
  ml::Vec y = {0, 0, 1, 1, 0, 1, 0, 1, 0, 1};
  // seed_rng fixes the forest member's bootstrap sampling.
  ml::seed_rng(1);
  course::VotingStacking model(course::VotingStacking::kSoft);
  model.fit(X, y);
  std::cout << "soft voting on toy data: " << ml::print_vec(model.predict(X))
            << "\n";
}
