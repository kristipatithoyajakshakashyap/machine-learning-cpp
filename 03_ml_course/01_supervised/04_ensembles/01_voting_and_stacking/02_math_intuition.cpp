// 03_ml_course/01_supervised/04_ensembles/01_voting_and_stacking/02_math_intuition.cpp
//
// Purpose : Lesson 2 - the arithmetic of the three combination rules on one
//           hand-written row of three member probability vectors: hard vote
//           counts, the soft average, and a member-weighted average standing
//           in for a stacking meta-learner. Ends with real vote fractions
//           from a hard-voting ensemble on a toy set.
// Inputs  : none (numbers and fixture are inline); no defines used.
// Outputs : prints only.
// Run     : target ens_math_intuition (no arguments).
#include "Model.hpp"
#include <iomanip>
#include <iostream>
// Prints the probability arithmetic behind the three modes on one toy row.
int main() {
  std::cout
      << std::setprecision(3)
      << R"LESSON(Three members, three classes, one test row. Hard voting: labels (0, 0, 2) -> class 0 with 2/3 of the votes. Soft voting: average the probability rows. Stacking: the meta-learner applies learned weights to the nine stacked probabilities instead of equal weights.)LESSON"
      << "\n\n";
  // p[b] = member b's probabilities over the 3 classes; argmaxes are 0, 0, 2.
  ml::Mat p = {{0.6, 0.3, 0.1}, {0.5, 0.4, 0.1}, {0.2, 0.3, 0.5}};
  ml::Vec avg(3, 0.0);
  for (const auto &row : p)
    for (size_t c = 0; c < 3; ++c)
      avg[c] += row[c] / 3.0;
  std::cout << "hard votes per class: 2 0 1 -> class 0\n"
            << "soft average: " << ml::print_vec(avg) << " -> class 0\n";
  // A meta-learner that has learned the forest is the most reliable member.
  ml::Vec w = {0.2, 0.2, 0.6};
  ml::Vec weighted(3, 0.0);
  for (size_t b = 0; b < 3; ++b)
    for (size_t c = 0; c < 3; ++c)
      weighted[c] += w[b] * p[b][c];
  // Only classes 0 and 2 can win here, so a two-way comparison suffices.
  std::cout << "member-weighted (0.2, 0.2, 0.6): " << ml::print_vec(weighted)
            << " -> class " << (weighted[0] >= weighted[2] ? 0 : 2) << "\n"
            << "Equal weights and learned weights can disagree; stacking earns "
               "its weights on out-of-fold data.\n";
  // Same toy set as lesson 1; vote fractions are multiples of 1/3.
  ml::Mat X = {{-2, 0}, {-1, 1}, {1, 0}, {2, 1}, {-3, 1},
               {3, 0}, {-1.5, 0}, {1.5, 1}, {-2.5, 1}, {2.5, 0}};
  ml::Vec y = {0, 0, 1, 1, 0, 1, 0, 1, 0, 1};
  ml::seed_rng(1);
  course::VotingStacking hard(course::VotingStacking::kHard);
  hard.fit(X, y);
  std::cout << "hard-vote fractions on first row: "
            << ml::print_vec(hard.predict_proba({X[0]})[0]) << "\n";
}
