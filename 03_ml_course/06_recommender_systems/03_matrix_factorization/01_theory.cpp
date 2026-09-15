#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Represent users and items by short latent vectors. A global mean, user bias and item bias account for broad rating tendencies; vector products capture interactions. Missing ratings contribute no squared-error term. This is explicit-feedback factorization, not an implicit confidence model.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
